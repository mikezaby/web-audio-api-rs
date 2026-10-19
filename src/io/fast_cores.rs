//! The fastest cores of an Android device, for the threads that render
//!
//! AAudio's low-latency path runs the output callback at real-time priority, and Android places
//! real-time threads on any core, mostly the efficiency cores. There a render took three to four
//! times longer than at normal priority, where the scheduler moves busy threads to the
//! performance cores (Galaxy Tab, 4 + 4 cores), and the audio fell behind. The render thread and
//! its helpers keep to the faster cores instead, as helpers join the device workgroup on macOS
//! (`workgroup.rs`).

#[cfg(target_os = "android")]
pub(crate) use android::{find, prefer};

/// Elsewhere the scheduler places real-time threads well
#[cfg(not(target_os = "android"))]
pub(crate) fn prefer() {}

#[cfg(target_os = "android")]
mod android {
    use std::cell::Cell;
    use std::sync::{Once, OnceLock};

    static FAST: OnceLock<Option<libc::cpu_set_t>> = OnceLock::new();

    /// Read which cores are faster than the slowest, once; on a control thread, as it reads files
    pub(crate) fn find() {
        FAST.get_or_init(fast_cores);
    }

    /// Keep the calling thread on the cores `find` found, once per thread, cheap after that
    ///
    /// Nothing happens before `find`, or on a device whose cores all run at the same speed.
    pub(crate) fn prefer() {
        thread_local! {
            static DONE: Cell<bool> = const { Cell::new(false) };
        }
        if DONE.replace(true) {
            return;
        }
        let Some(Some(set)) = FAST.get() else {
            return;
        };
        // SAFETY: `set` is a whole cpu set; pid 0 is the calling thread
        let moved =
            unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), set) };
        if moved != 0 {
            static WARNED: Once = Once::new();
            WARNED.call_once(|| log::warn!("Render threads could not keep to the fast cores"));
        }
    }

    /// Every core whose top clock is above the slowest cores', none when all are alike
    fn fast_cores() -> Option<libc::cpu_set_t> {
        // SAFETY: sysconf only reads a value
        let count = usize::try_from(unsafe { libc::sysconf(libc::_SC_NPROCESSORS_CONF) }).ok()?;
        let cores: Vec<(usize, u64)> = (0..count)
            .filter_map(|cpu| {
                let path = format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_max_freq");
                let khz = std::fs::read_to_string(path).ok()?.trim().parse().ok()?;
                Some((cpu, khz))
            })
            .collect();
        let slowest = cores.iter().map(|&(_, khz)| khz).min()?;

        // SAFETY: a zeroed cpu set is the empty set
        let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
        let mut any = false;
        for &(cpu, khz) in &cores {
            if khz > slowest {
                // SAFETY: `cpu` is below the core count, inside the set
                unsafe { libc::CPU_SET(cpu, &mut set) };
                any = true;
            }
        }
        if any {
            log::debug!("Render threads keep to the cores above {slowest} kHz");
        }
        any.then_some(set)
    }
}
