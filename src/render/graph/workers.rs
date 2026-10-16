//! Helper threads that render groups alongside the render thread, in the same quantum
//!
//! The scheduling follows Ardour's process graph. Each quantum, the groups that depend on no other
//! group go on a work queue. A thread that takes a group renders it, then counts down each group
//! that depends on it and queues the ones that have all their inputs now. The render thread takes
//! groups too, and returns once every group finished.

use std::cell::{Cell, UnsafeCell};
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Once};
use std::thread::JoinHandle;

use crossbeam_channel::{Receiver, Sender};

use super::groups::GroupPlan;
use super::Graph;
use crate::context::AudioNodeId;
use crate::events::EventDispatch;
use crate::io::workgroup::AudioWorkgroup;
use crate::render::{AudioWorkletGlobalScope, NodeCollection};

/// Queue entry that stops a helper thread
const STOP: usize = usize::MAX;

/// Most groups the work queue holds, a plan with more renders on the render thread alone
pub(super) const MAX_GROUPS: usize = 1024;

/// What the render thread publishes for the helpers before it queues the first group of a quantum
#[derive(Clone, Copy)]
struct Job {
    nodes: *const NodeCollection,
    plan: *const GroupPlan,
    current_frame: u64,
    current_time: f64,
}

struct Shared {
    job: UnsafeCell<Job>,
    /// Groups of this quantum that did not finish yet
    remaining: AtomicUsize,
}

// SAFETY: the render thread writes `job` only while no group of the quantum is queued or being
// rendered: before it queues the first group, and after `remaining` dropped to zero. A helper
// reads it only after it took a group from the queue, which orders the read after the write. The
// pointers in `job` are only followed while the quantum runs, while the render thread keeps the
// node collection and the plan alive and does not touch them mutably.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

/// How helpers of a real-time context run
#[derive(Clone, Debug)]
pub(crate) struct RealTime {
    /// Audio callback size in frames, for the real-time scheduling period
    pub frames: usize,
    /// Workgroup of the output device's IO thread, for the helpers to join
    pub workgroup: Option<Arc<AudioWorkgroup>>,
}

/// Helper render threads, installed with `Graph::set_workers`
pub(crate) struct Workers {
    queue: Sender<usize>,
    work: Receiver<usize>,
    shared: Arc<Shared>,
    handles: Vec<JoinHandle<()>>,
}

impl std::fmt::Debug for Workers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Workers")
            .field("helpers", &self.handles.len())
            .finish_non_exhaustive()
    }
}

impl Workers {
    /// Spawn `helpers` threads, which wait for work until dropped
    ///
    /// With `real_time`, the helpers ask for real-time priority and join the device workgroup,
    /// like the audio callback thread. Offline rendering passes `None`.
    ///
    /// # Panics
    ///
    /// Panics when the operating system cannot spawn a thread.
    pub(crate) fn spawn(
        helpers: usize,
        sample_rate: f32,
        event_sender: &Sender<EventDispatch>,
        real_time: Option<RealTime>,
    ) -> Self {
        let (queue, work) = crossbeam_channel::bounded(MAX_GROUPS + helpers);
        let shared = Arc::new(Shared {
            job: UnsafeCell::new(Job {
                nodes: std::ptr::null(),
                plan: std::ptr::null(),
                current_frame: 0,
                current_time: 0.,
            }),
            remaining: AtomicUsize::new(0),
        });

        let handles = (1..=helpers)
            .map(|i| {
                let queue = queue.clone();
                let work = work.clone();
                let shared = Arc::clone(&shared);
                let event_sender = event_sender.clone();
                let real_time = real_time.clone();
                std::thread::Builder::new()
                    .name(format!("web-audio-render-{i}"))
                    .spawn(move || {
                        let workgroup = real_time.as_ref().and_then(|rt| rt.workgroup.clone());
                        // keeps the membership until the helper exits
                        let _member = real_time.as_ref().map(|rt| {
                            promote_to_real_time(rt.frames, sample_rate);
                            join_workgroup(workgroup.as_deref())
                        });
                        helper(&queue, &work, &shared, sample_rate, event_sender);
                    })
                    .expect("Unable to spawn render thread")
            })
            .collect();

        Self {
            queue,
            work,
            shared,
            handles,
        }
    }

    /// Render every group of `plan`, on the helpers and this thread, return when all finished
    pub(super) fn render(
        &self,
        nodes: &NodeCollection,
        plan: &GroupPlan,
        scope: &AudioWorkletGlobalScope,
    ) {
        // SAFETY: no group of this quantum is queued yet, so no helper reads the job, see `Shared`
        unsafe {
            *self.shared.job.get() = Job {
                nodes,
                plan,
                current_frame: scope.current_frame,
                current_time: scope.current_time,
            };
        }
        for (pending, group) in plan.pending.iter().zip(&plan.groups) {
            pending.store(group.dependencies, Ordering::Relaxed);
        }
        self.shared
            .remaining
            .store(plan.groups.len(), Ordering::Relaxed);

        // `order` starts with every group that depends on no other group
        let roots = plan
            .order
            .iter()
            .take_while(|&&group| plan.groups[group].dependencies == 0);
        for &group in roots {
            // the queue has room for every group, so this does not block
            self.queue.send(group).unwrap();
        }

        // Work too, until the last group finished. Never block, this is the audio callback.
        let mut idle = 0_u32;
        while self.shared.remaining.load(Ordering::Acquire) > 0 {
            match self.work.try_recv() {
                Ok(group) => {
                    idle = 0;
                    work(
                        nodes,
                        plan,
                        group,
                        scope,
                        &self.queue,
                        &self.shared.remaining,
                    );
                }
                Err(_) => {
                    // ponytail: fixed spin then yield, tune on the Pi
                    idle += 1;
                    if idle < 64 {
                        std::hint::spin_loop();
                    } else {
                        std::thread::yield_now();
                    }
                }
            }
        }
    }
}

impl Drop for Workers {
    fn drop(&mut self) {
        for _ in &self.handles {
            let _ = self.queue.send(STOP);
        }
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

/// Render `group`, then queue each group that depends on it and has all its inputs now
fn work(
    nodes: &NodeCollection,
    plan: &GroupPlan,
    group: usize,
    scope: &AudioWorkletGlobalScope,
    queue: &Sender<usize>,
    remaining: &AtomicUsize,
) {
    // A panic of a processor is caught per node, so a panic here is a bug in the scheduler. Still
    // count the group down, so the quantum completes instead of the render thread waiting forever.
    let rendered = panic::catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: a group is queued once per quantum, after every group it depends on finished
        // (`pending` counts them down with AcqRel). So this thread alone works on it, and the
        // mailboxes it reads were written.
        unsafe { Graph::render_group(nodes, plan, group, scope) }
    }));
    if rendered.is_err() {
        log::error!("Render group {group} panicked");
    }

    for &dependent in &plan.groups[group].dependents {
        if plan.pending[dependent].fetch_sub(1, Ordering::AcqRel) == 1 {
            // the queue has room for every group, so this does not block
            queue.send(dependent).unwrap();
        }
    }
    remaining.fetch_sub(1, Ordering::AcqRel);
}

/// Give the calling helper real-time priority, or warn once and carry on at normal priority
fn promote_to_real_time(frames: usize, sample_rate: f32) {
    static WARNED: Once = Once::new();

    // buffer sizes and sample rates are far below u32::MAX
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let promoted = audio_thread_priority::promote_current_thread_to_real_time(
        frames as u32,
        sample_rate as u32,
    );
    match promoted {
        // the thread keeps the priority until it exits
        Ok(_) => log::debug!("Render helper runs at real-time priority"),
        Err(e) => WARNED.call_once(|| {
            log::warn!("Render helpers run at normal priority, real-time priority refused: {e}");
        }),
    }
}

/// Join the device workgroup with the calling helper, or warn once
///
/// Without it, macOS runs real-time helpers on efficiency cores.
fn join_workgroup(workgroup: Option<&AudioWorkgroup>) -> Option<impl Sized + '_> {
    static WARNED: Once = Once::new();

    let member = workgroup?.join();
    match member {
        Some(_) => log::debug!("Render helper joined the audio workgroup"),
        None => {
            WARNED.call_once(|| log::warn!("Render helpers could not join the audio workgroup"))
        }
    }
    member
}

/// Main loop of a helper thread
fn helper(
    queue: &Sender<usize>,
    work_queue: &Receiver<usize>,
    shared: &Shared,
    sample_rate: f32,
    event_sender: Sender<EventDispatch>,
) {
    let mut scope = AudioWorkletGlobalScope {
        current_frame: 0,
        current_time: 0.,
        sample_rate,
        node_id: Cell::new(AudioNodeId(0)),
        event_sender,
    };

    // Same floating point mode as the render thread, so any thread count renders the same samples
    #[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
    unsafe {
        // SAFETY: as for the render thread (`src/render/thread.rs`)
        no_denormals::no_denormals(|| helper_loop(queue, work_queue, shared, &mut scope));
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    helper_loop(queue, work_queue, shared, &mut scope);
}

fn helper_loop(
    queue: &Sender<usize>,
    work_queue: &Receiver<usize>,
    shared: &Shared,
    scope: &mut AudioWorkletGlobalScope,
) {
    // Sleep as soon as the queue is empty, beyond the short backoff inside `recv`. At real-time
    // priority, spinning counts against the time-constraint budget: on macOS at 64 voices per
    // track, a 50 us spin raised missed callbacks from 0.2% to 1.1%, a 3 ms spin to 34%.
    while let Ok(group) = work_queue.recv() {
        if group == STOP {
            break;
        }
        // SAFETY: the job of this quantum was published before the group was queued
        let job = unsafe { *shared.job.get() };
        scope.current_frame = job.current_frame;
        scope.current_time = job.current_time;
        // SAFETY: the quantum is running, see `Shared`
        let (nodes, plan) = unsafe { (&*job.nodes, &*job.plan) };
        work(nodes, plan, group, scope, queue, &shared.remaining);
    }
}
