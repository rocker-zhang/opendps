/// NVML cap enforcement backend (P2-M1).
///
/// Implements `PowerSource` + `CapSink` using `nvml-wrapper`.  This is the
/// real hardware path: `set_cap_w` calls
/// `nvmlDeviceSetPowerManagementLimit(handle, milliwatts)` directly.
///
/// The module is only compiled when the `nvml` feature is enabled; the binary
/// defaults to `sim` mode so CI can build without a GPU node.
#[cfg(feature = "nvml")]
pub mod nvml {
    use crate::failsafe::{CapSink, PowerSource};
    use nvml_wrapper::{Device, Nvml};
    use std::sync::Mutex;

    pub struct NvmlBackend {
        _nvml: Nvml,
        devices: Vec<Mutex<Device<'static>>>,
    }

    impl NvmlBackend {
        /// Initialise NVML and enumerate all devices.
        pub fn init() -> Result<Self, nvml_wrapper::error::NvmlError> {
            // SAFETY: Nvml::init() acquires the NVML context for the process lifetime.
            let nvml = Nvml::init()?;
            let count = nvml.device_count()? as usize;
            // We store Device<'static> by transmuting the lifetime to allow storage
            // alongside the Nvml instance.  This is safe because NvmlBackend owns
            // the Nvml context and devices are only accessed while NvmlBackend lives.
            let devices = (0..count)
                .map(|i| {
                    let d = nvml.device_by_index(i as u32)?;
                    // SAFETY: device lifetime is tied to the Nvml context we own.
                    let d: Device<'static> = unsafe { std::mem::transmute(d) };
                    Ok(Mutex::new(d))
                })
                .collect::<Result<Vec<_>, nvml_wrapper::error::NvmlError>>()?;

            Ok(Self {
                _nvml: nvml,
                devices,
            })
        }

        pub fn gpu_count(&self) -> usize {
            self.devices.len()
        }
    }

    // NVML is documented thread-safe for power management calls;
    // each Device is protected by a Mutex.
    unsafe impl Send for NvmlBackend {}
    unsafe impl Sync for NvmlBackend {}

    impl NvmlBackend {
        /// Maximum enforced power cap (hardware limit), in watts.
        pub fn max_cap_w(&self, gpu: usize) -> f64 {
            let d = self.devices[gpu].lock().unwrap();
            d.power_management_limit_constraints()
                .map(|c| c.max_limit as f64 / 1000.0)
                .unwrap_or(0.0)
        }

        /// Current active power management limit, in watts.
        pub fn current_cap_w(&self, gpu: usize) -> f64 {
            let d = self.devices[gpu].lock().unwrap();
            d.power_management_limit()
                .map(|mw| mw as f64 / 1000.0)
                .unwrap_or(0.0)
        }

        pub fn name(&self, gpu: usize) -> String {
            let d = self.devices[gpu].lock().unwrap();
            d.name().unwrap_or_else(|_| format!("GPU{gpu}"))
        }
    }

    impl PowerSource for NvmlBackend {
        fn power_draw_w(&self, gpu: usize) -> f64 {
            let d = self.devices[gpu].lock().unwrap();
            d.power_usage().map(|mw| mw as f64 / 1000.0).unwrap_or(0.0)
        }
        fn gpu_count(&self) -> usize {
            self.devices.len()
        }
    }

    impl CapSink for NvmlBackend {
        fn set_cap_w(&self, gpu: usize, w: f64) {
            let mut d = self.devices[gpu].lock().unwrap();
            let mw = (w * 1000.0) as u32;
            if let Err(e) = d.set_power_management_limit(mw) {
                tracing::error!(gpu, w, "set_power_management_limit failed: {e}");
            }
        }
    }

    // -----------------------------------------------------------------------
    // NvmlIpcAdapter — bridges Arc<NvmlBackend> into IpcBackend
    // -----------------------------------------------------------------------

    /// Wraps a shared `Arc<NvmlBackend>` so it can be passed to
    /// [`crate::ipc::spawn_ipc_listener`] inside an `Arc<Mutex<NvmlIpcAdapter>>`.
    /// The outer `Mutex` is only held for the duration of each IPC command;
    /// the inner per-device `Mutex`es in `NvmlBackend` are independent.
    pub struct NvmlIpcAdapter {
        inner: std::sync::Arc<NvmlBackend>,
    }

    impl NvmlIpcAdapter {
        pub fn new(backend: std::sync::Arc<NvmlBackend>) -> Self {
            Self { inner: backend }
        }
    }

    impl crate::ipc::IpcBackend for NvmlIpcAdapter {
        fn set_cap(&mut self, gpu: usize, watts: f64) {
            self.inner.set_cap_w(gpu, watts);
        }

        fn get_cap(&self, gpu: usize) -> f64 {
            self.inner.current_cap_w(gpu)
        }

        fn get_draw(&self, gpu: usize) -> f64 {
            self.inner.power_draw_w(gpu)
        }

        fn gpu_count(&self) -> usize {
            self.inner.gpu_count()
        }
    }
}
