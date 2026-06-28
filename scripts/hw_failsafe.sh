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
# Cargo workspace: binaries land in the workspace target dir, not the crate's.
AGENT_BIN="$ROOT/target/release/opendps-agent"

echo "=== DC4.1: failsafe loop latency benchmark (no GPU required) ==="
cargo bench --bench bench_failsafe 2>&1 | grep -iE "p50|p99|latency|time:|µs|us" || \
  echo "(bench produced no latency lines — see full output with: cargo bench)"

echo
echo "=== DC4.2: real NVML cap-lower trip ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPUs detected:"
  nvidia-smi --query-gpu=index,name,power.draw,power.limit,power.max_limit --format=csv,noheader
  echo "Building agent with NVML..."
  cargo build --release --features nvml

  # Derive the trip profile from the actual hardware rather than hard-coding one
  # set of watts for every GPU model: threshold just below the highest current
  # draw (so even an idle fleet trips), emergency cap = the device min limit.
  # Override either via OPENDPS_FAILSAFE_THRESHOLD_W / _CAP_W.
  MAX_DRAW=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | sort -n | tail -1)
  MIN_LIMIT=$(nvidia-smi --query-gpu=power.min_limit --format=csv,noheader,nounits | sort -n | head -1)
  THRESH="${OPENDPS_FAILSAFE_THRESHOLD_W:-$(awk "BEGIN{printf \"%d\", $MAX_DRAW - 5}")}"
  CAP="${OPENDPS_FAILSAFE_CAP_W:-$(awk "BEGIN{printf \"%d\", $MIN_LIMIT}")}"
  PORT="${OPENDPS_METRICS_PORT:-9403}"
  HOST="${OPENDPS_HOST:-127.0.0.1}"
  echo "Trip threshold=${THRESH}W (below max idle draw ${MAX_DRAW}W); emergency cap=${CAP}W (min limit ${MIN_LIMIT}W)."
  "$AGENT_BIN" \
    --nvml \
    --failsafe-threshold-w "$THRESH" \
    --failsafe-cap-w "$CAP" \
    --metrics-port "$PORT" &
  AGENT_PID=$!
  trap 'kill "$AGENT_PID" 2>/dev/null || true' INT TERM EXIT
  sleep 5
  echo "Failsafe metrics:"
  curl -s "$HOST:$PORT/metrics" 2>/dev/null | grep -iE "failsafe|trip|latency" | grep -v '^#' || true
  kill "$AGENT_PID" 2>/dev/null || true
  trap - INT TERM EXIT
  echo "NOTE: the failsafe lowered caps; restore per GPU with 'nvidia-smi -pl <default_watts>'."
else
  echo "No nvidia-smi on this host — skipping the real cap-lower trip."
  echo "Run on a cap-capable GPU node for the full DC4 check."
fi
