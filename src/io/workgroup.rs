//! The OS workgroup of an output device's IO thread (macOS)
//!
//! Helper render threads join the workgroup of the device the context plays to, so the scheduler
//! treats them like the device's IO thread, with its deadline. Without it, macOS runs real-time
//! helpers on efficiency cores, where a render group takes about 3.8 times longer (M1 Pro).

#[cfg(target_os = "macos")]
pub(crate) use macos::AudioWorkgroup;

/// Workgroups exist on macOS only, elsewhere there is never one to join
#[cfg(not(target_os = "macos"))]
#[derive(Debug)]
pub(crate) enum AudioWorkgroup {}

#[cfg(not(target_os = "macos"))]
impl AudioWorkgroup {
    pub(crate) fn join(&self) -> Option<()> {
        match *self {}
    }
}

#[cfg(target_os = "macos")]
mod macos {
    use std::ffi::c_void;
    use std::marker::PhantomData;
    use std::ptr::NonNull;

    #[repr(C)]
    struct AudioObjectPropertyAddress {
        selector: u32,
        scope: u32,
        element: u32,
    }

    /// `os_workgroup_join_token_s`
    #[repr(C)]
    struct JoinToken {
        sig: u32,
        opaque: [u8; 36],
    }

    const SYSTEM_OBJECT: u32 = 1;
    const UNKNOWN_OBJECT: u32 = 0;
    const SCOPE_GLOBAL: u32 = u32::from_be_bytes(*b"glob");
    const ELEMENT_MAIN: u32 = 0;
    const TRANSLATE_UID_TO_DEVICE: u32 = u32::from_be_bytes(*b"uidd");
    const IO_THREAD_OS_WORKGROUP: u32 = u32::from_be_bytes(*b"oswg");
    const UTF8: u32 = 0x0800_0100;

    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        fn CFStringCreateWithBytes(
            allocator: *const c_void,
            bytes: *const u8,
            length: isize,
            encoding: u32,
            external_representation: u8,
        ) -> *const c_void;
        fn CFRelease(object: *const c_void);
    }

    #[link(name = "CoreAudio", kind = "framework")]
    extern "C" {
        fn AudioObjectGetPropertyData(
            object: u32,
            address: *const AudioObjectPropertyAddress,
            qualifier_size: u32,
            qualifier: *const c_void,
            data_size: *mut u32,
            data: *mut c_void,
        ) -> i32;
    }

    extern "C" {
        fn os_workgroup_join(workgroup: *mut c_void, token: *mut JoinToken) -> i32;
        fn os_workgroup_leave(workgroup: *mut c_void, token: *mut JoinToken);
        fn os_release(object: *mut c_void);
    }

    /// An `os_workgroup_t`, released on drop
    #[derive(Debug)]
    pub(crate) struct AudioWorkgroup(NonNull<c_void>);

    // SAFETY: os_workgroup objects are thread safe, joining and leaving happen per thread
    unsafe impl Send for AudioWorkgroup {}
    unsafe impl Sync for AudioWorkgroup {}

    impl Drop for AudioWorkgroup {
        fn drop(&mut self) {
            // SAFETY: we own the +1 reference returned by CoreAudio
            unsafe { os_release(self.0.as_ptr()) };
        }
    }

    /// Membership of the calling thread, it leaves the workgroup on drop
    pub(crate) struct Membership<'a> {
        workgroup: &'a AudioWorkgroup,
        // boxed, so the token keeps its address between join and leave
        token: Box<JoinToken>,
        // leaving happens on the thread that joined
        _thread: PhantomData<*const ()>,
    }

    impl Drop for Membership<'_> {
        fn drop(&mut self) {
            // SAFETY: the token was filled in by a successful join on this thread
            unsafe { os_workgroup_leave(self.workgroup.0.as_ptr(), &mut *self.token) };
        }
    }

    impl AudioWorkgroup {
        /// The workgroup of the IO thread of the CoreAudio device with this UID
        #[cfg_attr(not(feature = "cpal"), allow(dead_code))]
        pub(crate) fn for_device_uid(uid: &str) -> Option<Self> {
            // SAFETY: plain CoreFoundation and CoreAudio calls with valid pointers, the results
            // are checked before use, and the CFString is released after the lookup
            unsafe {
                let uid = CFStringCreateWithBytes(
                    std::ptr::null(),
                    uid.as_ptr(),
                    uid.len() as isize,
                    UTF8,
                    0,
                );
                if uid.is_null() {
                    return None;
                }

                let mut device = UNKNOWN_OBJECT;
                let mut size = std::mem::size_of::<u32>() as u32;
                let address = AudioObjectPropertyAddress {
                    selector: TRANSLATE_UID_TO_DEVICE,
                    scope: SCOPE_GLOBAL,
                    element: ELEMENT_MAIN,
                };
                let status = AudioObjectGetPropertyData(
                    SYSTEM_OBJECT,
                    &address,
                    std::mem::size_of::<*const c_void>() as u32,
                    (&uid as *const *const c_void).cast(),
                    &mut size,
                    (&mut device as *mut u32).cast(),
                );
                CFRelease(uid);
                if status != 0 || device == UNKNOWN_OBJECT {
                    return None;
                }

                let mut workgroup: *mut c_void = std::ptr::null_mut();
                let mut size = std::mem::size_of::<*mut c_void>() as u32;
                let address = AudioObjectPropertyAddress {
                    selector: IO_THREAD_OS_WORKGROUP,
                    scope: SCOPE_GLOBAL,
                    element: ELEMENT_MAIN,
                };
                let status = AudioObjectGetPropertyData(
                    device,
                    &address,
                    0,
                    std::ptr::null(),
                    &mut size,
                    (&mut workgroup as *mut *mut c_void).cast(),
                );
                if status != 0 {
                    return None;
                }
                NonNull::new(workgroup).map(Self)
            }
        }

        /// Join with the calling thread, which should run at real-time priority already
        pub(crate) fn join(&self) -> Option<Membership<'_>> {
            let mut token = Box::new(JoinToken {
                sig: 0,
                opaque: [0; 36],
            });
            // SAFETY: a valid workgroup and a token that outlives the membership
            let status = unsafe { os_workgroup_join(self.0.as_ptr(), &mut *token) };
            (status == 0).then(|| Membership {
                workgroup: self,
                token,
                _thread: PhantomData,
            })
        }
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::AudioWorkgroup;

    #[test]
    fn unknown_device_has_no_workgroup() {
        assert!(AudioWorkgroup::for_device_uid("no such device").is_none());
    }

    #[cfg(feature = "cpal")]
    #[test]
    fn real_time_thread_joins_output_device_workgroup() {
        use cpal::traits::{DeviceTrait, HostTrait};

        let Some(device) = cpal::default_host().default_output_device() else {
            return; // no audio hardware
        };
        let uid = device.id().unwrap();
        let workgroup = AudioWorkgroup::for_device_uid(uid.id()).expect("device has a workgroup");

        let joined = std::thread::spawn(move || {
            audio_thread_priority::promote_current_thread_to_real_time(128, 48000).unwrap();
            let joined = workgroup.join().is_some();
            joined
        });
        assert!(joined.join().unwrap());
    }
}
