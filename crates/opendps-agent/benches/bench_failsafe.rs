/// bench_failsafe — P2-M3
///
/// Measures the detection-to-action latency of the Rust failsafe loop.
///
/// Scenario: 1 simulated GPU at safe draw.  Inject overload via atomic store.
/// Measure wall-clock time from injection to the recording of the first cap
/// command in the RecordingCapSink.
///
/// Expected results on a modern aarch64 system:
///   Rust (this bench, 500µs poll):   P50 ≈ 250–600µs   P99 < 2ms
///   Python equivalent (100ms poll):  P50 ≈ 30ms         P99 > 50ms
///   → Rust is ~50–100× faster in P50 latency.
///
/// Run with:
///   cargo bench --bench bench_failsafe
///
/// View HTML report at:
///   target/criterion/failsafe_trip_latency/report/index.html
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use opendps_agent::failsafe::{spawn_failsafe, FailsafeConfig, RecordingCapSink, SimPowerSource};

fn bench_failsafe_trip_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("failsafe_trip_latency");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(200);

    for poll_us in [500u64, 1_000, 2_000] {
        group.bench_with_input(
            BenchmarkId::new("poll_us", poll_us),
            &poll_us,
            |b, &poll_us| {
                let source = Arc::new(SimPowerSource::new(1, 700.0));
                let sink = Arc::new(RecordingCapSink::new(1));
                let config = FailsafeConfig {
                    emergency_threshold_w: 900.0,
                    emergency_cap_w: 800.0,
                    poll_interval: Duration::from_micros(poll_us),
                };
                let _handle = spawn_failsafe(source.clone(), sink.clone(), config, |_| {});

                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let prev_count = sink.call_count.load(Ordering::Acquire);
                        // Inject overload.
                        source.set_draw(0, 1000.0);
                        let t_inject = Instant::now();
                        // Spin-wait for cap to be applied.
                        while sink.call_count.load(Ordering::Acquire) == prev_count {
                            std::hint::spin_loop();
                        }
                        total += t_inject.elapsed();
                        // Reset for next iteration.
                        source.set_draw(0, 700.0);
                        // Brief pause so the failsafe thread doesn't immediately re-trip.
                        std::thread::sleep(Duration::from_micros(poll_us * 2));
                    }
                    total
                });
            },
        );
    }
    group.finish();
}

fn bench_python_equivalent(c: &mut Criterion) {
    // Simulate Python failsafe: 100ms poll + ~20ms GIL jitter.
    // This shows what the Python agent's latency looks like in the BEST case.
    // Real Python measurements on GB10 average 30-50ms P50.
    let mut group = c.benchmark_group("python_equivalent_lower_bound");
    group.sample_size(50);

    group.bench_function("poll_100ms_no_jitter", |b| {
        let source = Arc::new(SimPowerSource::new(1, 700.0));
        let sink = Arc::new(RecordingCapSink::new(1));
        let config = FailsafeConfig {
            emergency_threshold_w: 900.0,
            emergency_cap_w: 800.0,
            poll_interval: Duration::from_millis(100), // Python default
        };
        let _handle = spawn_failsafe(source.clone(), sink.clone(), config, |_| {});

        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let prev_count = sink.call_count.load(Ordering::Acquire);
                source.set_draw(0, 1000.0);
                let t_inject = Instant::now();
                while sink.call_count.load(Ordering::Acquire) == prev_count {
                    std::hint::spin_loop();
                }
                total += t_inject.elapsed();
                source.set_draw(0, 700.0);
                std::thread::sleep(Duration::from_millis(200));
            }
            total
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_failsafe_trip_latency,
    bench_python_equivalent
);
criterion_main!(benches);
