#!/usr/bin/env bash
# opendps N7 DC4 — real-GPU failsafe latency check.
#
# Run this ON a cap-capable GPU node (A10 / B300 / GB200). It:
#   1. Benchmarks the brain-independent failsafe detection loop (no GPU needed).
#   2. If a GPU + NVML are present, builds the agent with --features nvml and
#      runs a real cap-lower trip, reporting the round-trip latency.
#
# The failsafe is cap-lower-only and brain-independent: an injected overload
# must trip it and lower the cap in well under 1 ms at the loop level.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/crates/opendps-agent"

echo "=== DC4.1: failsafe loop latency benchmark (no GPU required) ==="
cargo bench --bench bench_failsafe 2>&1 | grep -iE "p50|p99|latency|time:|µs|us" || \
  echo "(bench produced no latency lines — see full output with: cargo bench)"

echo
echo "=== DC4.2: real NVML cap-lower trip ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU detected:"; nvidia-smi --query-gpu=index,name,power.limit --format=csv,noheader
  echo "Building agent with NVML..."
  cargo build --release --features nvml
  echo "Running failsafe with a threshold below idle draw to force a trip."
  echo "(Set OPENDPS_FAILSAFE_THRESHOLD_W / OPENDPS_FAILSAFE_CAP_W to tune.)"
  ./target/release/opendps-agent \
    --nvml \
    --failsafe-threshold-w "${OPENDPS_FAILSAFE_THRESHOLD_W:-220}" \
    --failsafe-cap-w "${OPENDPS_FAILSAFE_CAP_W:-200}" \
    --metrics-port 9403 &
  AGENT_PID=$!
  sleep 5
  echo "Failsafe metrics:"
  curl -s localhost:9403/metrics 2>/dev/null | grep -iE "failsafe|trip|latency" | grep -v '^#' || true
  kill "$AGENT_PID" 2>/dev/null || true
  echo "Restore caps with: nvidia-smi -pl <default_watts>"
else
  echo "No nvidia-smi on this host — skipping the real cap-lower trip."
  echo "Run on an A10/B300/GB200 node for the full DC4 check."
fi
