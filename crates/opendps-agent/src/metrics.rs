/// Prometheus metrics exported by opendps-agent (Rust).
///
/// Exposes a minimal /metrics HTTP endpoint using std::net::TcpListener
/// (zero extra dependencies).
///
/// Metrics:
///   opendps_agent_failsafe_trip_total{gpu}       — cumulative trip count
///   opendps_agent_failsafe_latency_us_bucket     — latency histogram (µs)
///   opendps_agent_power_cap_watts{gpu}           — current enforced cap
///   opendps_agent_power_draw_watts{gpu}          — last observed draw
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

pub struct LatencyHistogram {
    buckets: Vec<(u64, AtomicU64)>, // (upper_bound_us, count)
    sum_us: AtomicU64,
    count: AtomicU64,
}

impl LatencyHistogram {
    pub fn new() -> Self {
        let bounds: Vec<u64> = vec![50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 50_000];
        Self {
            buckets: bounds.into_iter().map(|b| (b, AtomicU64::new(0))).collect(),
            sum_us: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    pub fn observe_us(&self, latency_us: u64) {
        self.sum_us.fetch_add(latency_us, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        for (bound, cnt) in &self.buckets {
            if latency_us <= *bound {
                cnt.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Render as Prometheus text exposition format.
    pub fn render_prometheus(&self, name: &str) -> String {
        let mut out = format!(
            "# HELP {name} Failsafe trip detection-to-action latency in microseconds\n\
             # TYPE {name} histogram\n"
        );
        for (bound, cnt) in &self.buckets {
            out.push_str(&format!(
                "{name}_bucket{{le=\"{bound}\"}} {}\n",
                cnt.load(Ordering::Relaxed)
            ));
        }
        out.push_str(&format!(
            "{name}_bucket{{le=\"+Inf\"}} {}\n",
            self.count.load(Ordering::Relaxed)
        ));
        out.push_str(&format!(
            "{name}_sum {}\n",
            self.sum_us.load(Ordering::Relaxed)
        ));
        out.push_str(&format!(
            "{name}_count {}\n",
            self.count.load(Ordering::Relaxed)
        ));
        out
    }
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new()
    }
}

pub struct AgentMetrics {
    pub failsafe_latency: Arc<LatencyHistogram>,
    pub failsafe_trips: AtomicU64,
    pub power_draws: Mutex<Vec<f64>>,
    pub power_caps: Mutex<Vec<f64>>,
    pub n_gpus: usize,
}

impl AgentMetrics {
    pub fn new(n_gpus: usize) -> Arc<Self> {
        Arc::new(Self {
            failsafe_latency: Arc::new(LatencyHistogram::new()),
            failsafe_trips: AtomicU64::new(0),
            power_draws: Mutex::new(vec![0.0; n_gpus]),
            power_caps: Mutex::new(vec![0.0; n_gpus]),
            n_gpus,
        })
    }

    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(
            &self
                .failsafe_latency
                .render_prometheus("opendps_agent_failsafe_latency_us"),
        );
        out.push_str(&format!(
            "# HELP opendps_agent_failsafe_trip_total Cumulative failsafe trips\n\
             # TYPE opendps_agent_failsafe_trip_total counter\n\
             opendps_agent_failsafe_trip_total {}\n",
            self.failsafe_trips.load(Ordering::Relaxed)
        ));
        let draws = self.power_draws.lock().unwrap();
        let caps = self.power_caps.lock().unwrap();
        for gpu in 0..self.n_gpus {
            out.push_str(&format!(
                "opendps_agent_power_draw_watts{{gpu=\"{gpu}\"}} {}\n",
                draws[gpu]
            ));
            out.push_str(&format!(
                "opendps_agent_power_cap_watts{{gpu=\"{gpu}\"}} {}\n",
                caps[gpu]
            ));
        }
        out
    }
}

/// Serve /metrics on `addr` (e.g. "0.0.0.0:9403") in a background thread.
pub fn serve_metrics(addr: &str, metrics: Arc<AgentMetrics>) {
    let listener = std::net::TcpListener::bind(addr)
        .unwrap_or_else(|e| panic!("cannot bind metrics server to {addr}: {e}"));
    tracing::info!("Prometheus /metrics server on {addr}");
    std::thread::Builder::new()
        .name("metrics-server".into())
        .spawn(move || {
            for stream in listener.incoming() {
                match stream {
                    Ok(mut s) => {
                        let body = metrics.render();
                        let response = format!(
                            "HTTP/1.1 200 OK\r\n\
                             Content-Type: text/plain; version=0.0.4\r\n\
                             Content-Length: {}\r\n\
                             Connection: close\r\n\r\n\
                             {}",
                            body.len(),
                            body
                        );
                        use std::io::Write;
                        let _ = s.write_all(response.as_bytes());
                    }
                    Err(_) => break,
                }
            }
        })
        .expect("metrics server thread spawn failed");
}
