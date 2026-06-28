/// TCP IPC listener for the opendps-agent (P2-M3).
///
/// Accepts one or more concurrent connections on 127.0.0.1:9500.
/// Each connection speaks a newline-delimited JSON protocol used by the
/// Python AgentBridge:
///
/// ```text
/// {"cmd": "set_cap",  "gpu": 0, "watts": 850.0}  → no response
/// {"cmd": "get_draws"}                             → {"draws": {"0": 237.4, ...}}
/// {"cmd": "get_caps"}                              → {"caps": {"0": 1100.0, ...}}
/// ```
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;

use serde_json::{json, Value};
use tracing::{debug, error, info, warn};

use crate::failsafe::{CapSink, PowerSource, RecordingCapSink, SimPowerSource};

// ---------------------------------------------------------------------------
// IpcBackend trait
// ---------------------------------------------------------------------------

/// Abstraction over the GPU backend used by the IPC layer.
/// Implemented by [`SimIpcBackend`] (sim mode) and, under `feature = "nvml"`,
/// by `NvmlIpcAdapter` defined in `nvml_backend.rs`.
pub trait IpcBackend: Send + 'static {
    fn set_cap(&mut self, gpu: usize, watts: f64);
    fn get_cap(&self, gpu: usize) -> f64;
    fn get_draw(&self, gpu: usize) -> f64;
    fn gpu_count(&self) -> usize;
}

// ---------------------------------------------------------------------------
// Sim-mode backend
// ---------------------------------------------------------------------------

/// Bridges the split sim backend (separate source + sink Arcs) into a single
/// `IpcBackend` implementation.
pub struct SimIpcBackend {
    source: Arc<SimPowerSource>,
    sink: Arc<RecordingCapSink>,
}

impl SimIpcBackend {
    pub fn new(source: Arc<SimPowerSource>, sink: Arc<RecordingCapSink>) -> Self {
        Self { source, sink }
    }
}

impl IpcBackend for SimIpcBackend {
    fn set_cap(&mut self, gpu: usize, watts: f64) {
        self.sink.set_cap_w(gpu, watts);
    }

    fn get_cap(&self, gpu: usize) -> f64 {
        self.sink.last_cap_w(gpu)
    }

    fn get_draw(&self, gpu: usize) -> f64 {
        self.source.power_draw_w(gpu)
    }

    fn gpu_count(&self) -> usize {
        self.source.gpu_count()
    }
}

// ---------------------------------------------------------------------------
// Listener + connection handler
// ---------------------------------------------------------------------------

/// Spawns a background thread that binds to `127.0.0.1:{port}` and handles
/// inbound IPC connections.  Each connection is handled on its own thread.
/// The function returns immediately; the listener thread runs for the process
/// lifetime.
pub fn spawn_ipc_listener<B: IpcBackend>(backend: Arc<Mutex<B>>, port: u16) {
    thread::Builder::new()
        .name("ipc-listener".into())
        .spawn(move || {
            let addr = format!("127.0.0.1:{port}");
            let listener = match TcpListener::bind(&addr) {
                Ok(l) => {
                    info!("IPC listener on {addr}");
                    l
                }
                Err(e) => {
                    error!("IPC bind failed on {addr}: {e}");
                    return;
                }
            };
            for stream in listener.incoming() {
                match stream {
                    Ok(s) => {
                        let b = Arc::clone(&backend);
                        thread::spawn(move || handle_conn(s, b));
                    }
                    Err(e) => warn!("IPC accept error: {e}"),
                }
            }
        })
        .expect("ipc-listener thread spawn failed");
}

fn handle_conn<B: IpcBackend>(stream: TcpStream, backend: Arc<Mutex<B>>) {
    let peer = stream.peer_addr().ok();
    debug!("IPC connection from {peer:?}");
    let mut writer = stream.try_clone().expect("clone IPC stream");
    let reader = BufReader::new(stream);

    for line in reader.lines() {
        let line = match line {
            Ok(l) if !l.trim().is_empty() => l,
            Ok(_) => continue, // blank line — skip
            Err(_) => break,   // EOF or read error — close connection
        };
        let cmd: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                warn!("IPC JSON parse error ({peer:?}): {e} — line: {line:?}");
                continue;
            }
        };
        if let Some(resp) = dispatch(&cmd, &backend) {
            if writeln!(writer, "{resp}").is_err() {
                break;
            }
        }
    }
    debug!("IPC connection from {peer:?} closed");
}

fn dispatch<B: IpcBackend>(cmd: &Value, backend: &Arc<Mutex<B>>) -> Option<String> {
    let op = cmd["cmd"].as_str()?;
    let mut b = backend.lock().ok()?;
    match op {
        "set_cap" => {
            let gpu = cmd["gpu"].as_u64()? as usize;
            let watts = cmd["watts"].as_f64()?;
            b.set_cap(gpu, watts);
            None // no response needed
        }
        "get_draws" => {
            let n = b.gpu_count();
            let draws: serde_json::Map<String, Value> = (0..n)
                .map(|i| (i.to_string(), json!(b.get_draw(i))))
                .collect();
            Some(json!({"draws": draws}).to_string())
        }
        "get_caps" => {
            let n = b.gpu_count();
            let caps: serde_json::Map<String, Value> = (0..n)
                .map(|i| (i.to_string(), json!(b.get_cap(i))))
                .collect();
            Some(json!({"caps": caps}).to_string())
        }
        unknown => {
            warn!("Unknown IPC cmd: {unknown:?}");
            None
        }
    }
}
