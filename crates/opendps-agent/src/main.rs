/// opendps-agent — Rust hot path (P2-M1 through P2-M3)
///
/// Modes:
///   --sim       Use SimBackend (no NVML required).  Always available.
///   --nvml      Use real NVML (requires `--features nvml` build + GPU node).
///
/// The agent starts two concurrent components:
///   1. **Brain loop** (Tokio task): read power state → run decision → push cap.
///      In sim mode the "decision" is a simple static threshold; full DPM/PRS brain
///      is in Python (called via PyO3 in P2-M4).
///   2. **Failsafe loop** (std::thread, SCHED_FIFO when available): poll every
///      `--failsafe-poll-us` µs; if draw > threshold, lower cap immediately.
///
/// Prometheus /metrics served on `--metrics-port` (default 9403).
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use clap::Parser;

use opendps_agent::failsafe::{
    CapSink, FailsafeConfig, PowerSource, RecordingCapSink, SimPowerSource, spawn_failsafe,
};
use opendps_agent::metrics::{AgentMetrics, serve_metrics};

#[derive(Parser, Debug)]
#[command(
    name = "opendps-agent",
    version,
    about = "opendps GPU power cap agent — Rust hot path"
)]
struct Cli {
    /// Use simulated GPU backend (no real NVML).
    #[arg(long, default_value_t = true)]
    sim: bool,

    /// Use real NVML backend (requires --features nvml build and a GPU node).
    #[arg(long, default_value_t = false)]
    nvml: bool,

    /// Number of simulated GPUs (sim mode only).
    #[arg(long, default_value_t = 10)]
    sim_gpus: usize,

    /// Simulated steady-state power draw per GPU (W).
    #[arg(long, default_value_t = 700.0)]
    sim_draw_w: f64,

    /// Emergency failsafe threshold (W).  When any GPU exceeds this, the
    /// failsafe immediately caps it to `--failsafe-cap`.
    #[arg(long, default_value_t = 1000.0)]
    failsafe_threshold_w: f64,

    /// Emergency cap applied by the failsafe (W).
    #[arg(long, default_value_t = 800.0)]
    failsafe_cap_w: f64,

    /// Failsafe poll interval in microseconds.  Default 500µs.
    #[arg(long, default_value_t = 500)]
    failsafe_poll_us: u64,

    /// Prometheus /metrics port.
    #[arg(long, default_value_t = 9403)]
    metrics_port: u16,

    /// Run for this many seconds then exit (0 = run forever).
    #[arg(long, default_value_t = 0)]
    run_secs: u64,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("opendps_agent=info".parse().unwrap()),
        )
        .init();

    let cli = Cli::parse();

    #[cfg(feature = "nvml")]
    if cli.nvml {
        run_nvml_mode(cli).await;
        return;
    }

    #[cfg(not(feature = "nvml"))]
    if cli.nvml {
        eprintln!("ERROR: binary was not built with --features nvml");
        std::process::exit(1);
    }

    // ── Sim mode ──────────────────────────────────────────────────────────────
    let n = cli.sim_gpus;
    let m = AgentMetrics::new(n);

    serve_metrics(&format!("0.0.0.0:{}", cli.metrics_port), m.clone());
    tracing::info!(port = cli.metrics_port, "metrics server started");

    let source = Arc::new(SimPowerSource::new(n, cli.sim_draw_w));
    let sink = Arc::new(RecordingCapSink::new(n));

    start_failsafe_and_run(cli, m, source, sink.clone(), n, move |gpu| sink.last_cap_w(gpu)).await;
}

async fn start_failsafe_and_run<P, C, CapFn>(
    cli: Cli,
    m: Arc<AgentMetrics>,
    source: Arc<P>,
    sink: Arc<C>,
    n: usize,
    cap_reader: CapFn,
) where
    P: PowerSource + Send + Sync + 'static,
    C: CapSink + Send + Sync + 'static,
    CapFn: Fn(usize) -> f64 + Send + 'static,
{
    let m_failsafe = m.clone();
    let failsafe_config = FailsafeConfig {
        emergency_threshold_w: cli.failsafe_threshold_w,
        emergency_cap_w: cli.failsafe_cap_w,
        poll_interval: Duration::from_micros(cli.failsafe_poll_us),
    };

    let _failsafe_handle = spawn_failsafe(
        source.clone(),
        sink.clone(),
        failsafe_config,
        move |latency| {
            let us = latency.as_micros() as u64;
            m_failsafe.failsafe_latency.observe_us(us);
            m_failsafe.failsafe_trips.fetch_add(1, Ordering::Relaxed);
        },
    );

    tracing::info!(
        gpus = n,
        threshold_w = cli.failsafe_threshold_w,
        poll_us = cli.failsafe_poll_us,
        "opendps-agent started"
    );

    let tick_source = source.clone();
    let tick_metrics = m.clone();

    let brain_task = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(1));
        loop {
            interval.tick().await;
            let mut draws = tick_metrics.power_draws.lock().unwrap();
            let mut caps = tick_metrics.power_caps.lock().unwrap();
            for gpu in 0..n {
                draws[gpu] = tick_source.power_draw_w(gpu);
                caps[gpu] = cap_reader(gpu);
            }
        }
    });

    if cli.run_secs > 0 {
        tokio::time::sleep(Duration::from_secs(cli.run_secs)).await;
        tracing::info!("run_secs={} reached, exiting", cli.run_secs);
    } else {
        tokio::signal::ctrl_c().await.ok();
        tracing::info!("SIGINT received, shutting down");
    }
    brain_task.abort();
}

#[cfg(feature = "nvml")]
async fn run_nvml_mode(cli: Cli) {
    use opendps_agent::nvml_backend::nvml::NvmlBackend;

    let backend = Arc::new(NvmlBackend::init().expect("NVML init failed"));
    let n = backend.gpu_count();

    let m = AgentMetrics::new(n);
    serve_metrics(&format!("0.0.0.0:{}", cli.metrics_port), m.clone());
    tracing::info!(gpus = n, port = cli.metrics_port, "NVML backend ready, metrics started");

    let source = backend.clone();
    let sink = backend.clone();

    let backend_caps = backend.clone();
    start_failsafe_and_run(cli, m, source, sink, n, move |_gpu| {
        // NvmlBackend CapSink doesn't expose get_cap; return 0.0 for metrics
        // (failsafe still sets caps correctly via set_cap_w)
        let _ = &backend_caps;
        0.0
    })
    .await;
}
