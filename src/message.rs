//! Message passing from control to render node

use std::any::Any;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, TryLockError};

#[cfg(feature = "diagnostics")]
use crate::context::AudioBackendDiagnostics;
use crate::context::AudioNodeId;
use crate::node::{ChannelConfigInner, ChannelCountMode, ChannelInterpretation};
use crate::render::graph::workers::Workers;
use crate::render::graph::Graph;
use crate::render::AudioProcessor;

/// Commands from the control thread to the render thread
pub(crate) enum ControlMessage {
    /// Register a new node in the audio graph
    RegisterNode {
        id: AudioNodeId,
        reclaim_id: llq::Node<AudioNodeId>,
        node: Box<dyn AudioProcessor>,
        inputs: usize,
        outputs: usize,
        channel_config: ChannelConfigInner,
        render_group: u32,
    },

    /// Connect a node to another in the audio graph
    ConnectNode {
        from: AudioNodeId,
        to: AudioNodeId,
        input: usize,
        output: usize,
    },

    /// Clear the connection between two given nodes in the audio graph
    DisconnectNode {
        from: AudioNodeId,
        to: AudioNodeId,
        input: usize,
        output: usize,
    },

    /// Notify the render thread this node is dropped in the control thread
    ControlHandleDropped { id: AudioNodeId },

    /// Mark node as a cycle breaker (DelayNode only)
    MarkCycleBreaker { id: AudioNodeId },

    /// Shut down and recycle the audio graph
    CloseAndRecycle {
        sender: crossbeam_channel::Sender<Graph>,
    },

    /// Start rendering with given audio graph
    Startup { graph: Graph },

    /// Suspend and pause audio processing
    Suspend { notify: OneshotNotify },

    /// Resume audio processing after suspending
    Resume { notify: OneshotNotify },

    /// Stop audio processing
    Close { notify: OneshotNotify },

    /// Generic message to be handled by AudioProcessor
    NodeMessage {
        id: AudioNodeId,
        msg: llq::Node<Box<dyn Any + Send>>,
    },

    /// Request a diagnostic report of the audio graph
    #[cfg(feature = "diagnostics")]
    RunDiagnostics { backend: AudioBackendDiagnostics },

    /// Update the channel count of a node
    SetChannelCount { id: AudioNodeId, count: usize },

    /// Update the channel count mode of a node
    SetChannelCountMode {
        id: AudioNodeId,
        mode: ChannelCountMode,
    },

    /// Update the channel interpretation of a node
    SetChannelInterpretation {
        id: AudioNodeId,
        interpretation: ChannelInterpretation,
    },

    /// Render groups on helper threads too, or on the render thread alone with `None`
    SetRenderWorkers { workers: Option<Box<Workers>> },
}

/// Helper object to emit single notification
pub(crate) enum OneshotNotify {
    /// A synchronous oneshot sender
    Sync(crossbeam_channel::Sender<()>),
    /// An asynchronous oneshot sender
    Async(futures_channel::oneshot::Sender<()>),
}

impl OneshotNotify {
    /// Emit the notification
    pub fn send(self) {
        match self {
            Self::Sync(s) => s.send(()).ok(),
            Self::Async(s) => s.send(()).ok(),
        };
    }
}

/// The control thread's side of [`control_channel`]
#[derive(Clone)]
pub(crate) struct ControlSender {
    shared: Arc<ControlShared>,
}

/// The render thread's side of [`control_channel`]
pub(crate) struct ControlReceiver {
    shared: Arc<ControlShared>,
    /// Messages taken from the senders' queue, oldest first. In a `Mutex` only so the receiver
    /// is `Sync`, as a context holding one must be; the render thread reaches it through
    /// `get_mut`, without locking
    taken: Mutex<VecDeque<ControlMessage>>,
}

impl std::fmt::Debug for ControlSender {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ControlSender").finish_non_exhaustive()
    }
}

impl std::fmt::Debug for ControlReceiver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ControlReceiver").finish_non_exhaustive()
    }
}

struct ControlShared {
    queue: Mutex<VecDeque<ControlMessage>>,
    receivers: AtomicUsize,
}

/// A message no render thread will take: every receiver is gone
#[derive(Debug)]
pub(crate) struct SendError;

/// Control messages to the render thread, which never waits for a sender
///
/// In a lock-free channel such as crossbeam's, a receiver waits out a send in progress: a sender
/// that claimed a slot and was preempted before writing it held the render thread for as long
/// as it was off the CPU, 22 ms once on a Cortex-A72. Here senders queue under a lock and the
/// render thread only tries the lock: while a sender holds it, the render thread takes the
/// messages with its next drain. It swaps its emptied queue for the senders' one, so it neither
/// allocates nor frees.
pub(crate) fn control_channel() -> (ControlSender, ControlReceiver) {
    let shared = Arc::new(ControlShared {
        queue: Mutex::new(VecDeque::with_capacity(CONTROL_QUEUE_CAPACITY)),
        receivers: AtomicUsize::new(1),
    });
    let receiver = ControlReceiver {
        shared: Arc::clone(&shared),
        taken: Mutex::new(VecDeque::with_capacity(CONTROL_QUEUE_CAPACITY)),
    };
    (ControlSender { shared }, receiver)
}

const CONTROL_QUEUE_CAPACITY: usize = 4096;

fn lock_queue(shared: &ControlShared) -> MutexGuard<'_, VecDeque<ControlMessage>> {
    shared.queue.lock().unwrap_or_else(PoisonError::into_inner)
}

impl ControlSender {
    pub(crate) fn send(&self, msg: ControlMessage) -> Result<(), SendError> {
        if self.shared.receivers.load(Ordering::Acquire) == 0 {
            return Err(SendError);
        }
        lock_queue(&self.shared).push_back(msg);
        Ok(())
    }
}

impl ControlReceiver {
    /// The oldest message, or none while there is none or a sender holds the queue
    pub(crate) fn try_recv(&mut self) -> Option<ControlMessage> {
        let taken = self.taken.get_mut().unwrap_or_else(PoisonError::into_inner);
        if taken.is_empty() {
            let mut queue = match self.shared.queue.try_lock() {
                Ok(queue) => queue,
                Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
                Err(TryLockError::WouldBlock) => return None,
            };
            std::mem::swap(&mut *queue, taken);
        }
        taken.pop_front()
    }

    /// The messages no receiver has taken yet, for the control thread when it hands them to
    /// another render thread; the ones a render thread has taken it handles itself
    pub(crate) fn try_iter(&self) -> impl Iterator<Item = ControlMessage> {
        std::mem::take(&mut *lock_queue(&self.shared)).into_iter()
    }
}

impl Clone for ControlReceiver {
    fn clone(&self) -> Self {
        self.shared.receivers.fetch_add(1, Ordering::AcqRel);
        Self {
            shared: Arc::clone(&self.shared),
            taken: Mutex::new(VecDeque::new()),
        }
    }
}

impl Drop for ControlReceiver {
    fn drop(&mut self) {
        self.shared.receivers.fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dropped() -> ControlMessage {
        ControlMessage::ControlHandleDropped { id: AudioNodeId(1) }
    }

    #[test]
    fn test_receives_in_order_what_the_senders_queued() {
        let (sender, mut receiver) = control_channel();
        let other = sender.clone();
        sender.send(dropped()).unwrap();
        other
            .send(ControlMessage::MarkCycleBreaker { id: AudioNodeId(2) })
            .unwrap();

        assert!(matches!(
            receiver.try_recv(),
            Some(ControlMessage::ControlHandleDropped { .. })
        ));
        assert!(matches!(
            receiver.try_recv(),
            Some(ControlMessage::MarkCycleBreaker { .. })
        ));
        assert!(receiver.try_recv().is_none());
    }

    #[test]
    fn test_does_not_wait_for_a_sender_holding_the_queue() {
        let (sender, mut receiver) = control_channel();
        sender.send(dropped()).unwrap();

        // a sender preempted in the middle of a send
        let held = lock_queue(&sender.shared);
        assert!(receiver.try_recv().is_none());
        drop(held);

        assert!(matches!(
            receiver.try_recv(),
            Some(ControlMessage::ControlHandleDropped { .. })
        ));
    }

    #[test]
    fn test_receiving_does_not_allocate() {
        let (sender, mut receiver) = control_channel();
        // a first round, so both queues have their room
        (0..8).for_each(|_| sender.send(dropped()).unwrap());
        while receiver.try_recv().is_some() {}

        (0..8).for_each(|_| sender.send(dropped()).unwrap());
        alloc_counter::deny_alloc(|| {
            let mut received = 0;
            while let Some(msg) = receiver.try_recv() {
                std::mem::forget(msg);
                received += 1;
            }
            assert_eq!(received, 8);
        });
    }

    #[test]
    fn test_sending_fails_once_every_receiver_is_gone() {
        let (sender, receiver) = control_channel();
        let clone = receiver.clone();
        drop(receiver);
        assert!(sender.send(dropped()).is_ok());
        drop(clone);
        assert!(sender.send(dropped()).is_err());
    }
}
