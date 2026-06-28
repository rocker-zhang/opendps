#!/usr/bin/env bash
# opendps N7 — reproducible demo / acceptance check.
#
# Runs the demo done-criteria that need no GPU hardware (DC1-DC3, DC5-DC7) and
# asserts each. DC4 (real GPU failsafe latency) requires a cap-capable node and
# is gated — see scripts/hw_failsafe.sh.
#
# Usage:  ./scripts/demo.sh            # run all sim-completable criteria
#         SKIP_K8S=1 ./scripts/demo.sh # skip DC5 (no kind/k3s cluster)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG="deploy/topology-demo.json"
PASS=0; FAIL=0
ts() { date +%s; }
START=$(ts)

say()  { printf '\n=== %s ===\n' "$1"; }
ok()   { printf '  \033[32mPASS\033[0m %s\n' "$1"; PASS=$((PASS+1)); }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; FAIL=$((FAIL+1)); }
gate() { printf '  \033[33mGATED\033[0m %s\n' "$1"; }

# Resolve the controller entry point (console script or module).
if command -v opendps-controller >/dev/null 2>&1; then
  CTL=(opendps-controller)
else
  CTL=(python -m opendps.controller.standalone)
fi

# Run the controller in sim for a few seconds and print idle_stranded_watts.
stranded_for() {
  local brain=$1 port=$2 log pid val
  log="$(mktemp)"
  "${CTL[@]}" --sim --brain "$brain" --config "$CONFIG" \
    --metrics-port "$port" --interval 0.5 >"$log" 2>&1 &
  pid=$!
  # Ensure the background controller is killed even on Ctrl+C / early exit.
  trap 'kill "$pid" 2>/dev/null || true' INT TERM EXIT
  sleep 4  # let EWMA / solver converge
  val=$(curl -s "localhost:$port/metrics" 2>/dev/null \
        | awk '/^opendps_idle_stranded_watts\{/ {print $2}')
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  trap - INT TERM EXIT
  # Emit only a numeric value; empty/non-numeric becomes NaN (caught by caller).
  if [[ "$val" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then echo "$val"; else echo "NaN"; fi
}

is_num() { [[ "$1" =~ ^[0-9]+(\.[0-9]+)?$ ]]; }

# DC1 — single-workstation stack comes up.
say "DC1: sim stack reachable"
if curl -sf localhost:9090/-/healthy >/dev/null 2>&1; then
  ok "Prometheus healthy on :9090"
else
  gate "Prometheus not up (run: docker compose -f deploy/compose.yml up -d)"
fi

# DC2 — 10 GPUs in an 8-GPU budget: DPM strands power, PRS reclaims it.
say "DC2: stranded-watts reclaim (10 GPU / 8000 W)"
DPM=$(stranded_for dpm 19420)
PRS=$(stranded_for prs 19421)
if ! is_num "$DPM" || ! is_num "$PRS"; then
  bad "controller produced no stranded-watts metric (DPM=$DPM PRS=$PRS) — port in use or startup error"
else
  printf "  DPM baseline stranded = %.0f W\n  PRS  reclaimed stranded = %.0f W\n" "$DPM" "$PRS"
  if awk "BEGIN{exit !($DPM > 1000)}"; then ok "DPM baseline strands >1000 W ($DPM)"; else bad "DPM baseline too low ($DPM)"; fi
  if awk "BEGIN{exit !($PRS < $DPM*0.3)}"; then
    RECLAIM=$(awk "BEGIN{printf \"%.0f\", (1-$PRS/$DPM)*100}")
    ok "PRS reclaims ${RECLAIM}% of stranded watts"
  else
    bad "PRS did not reclaim enough ($PRS vs $DPM)"
  fi

  # DC3 — PRS on/off toggle produces a visibly different allocation.
  say "DC3: PRS on/off toggle"
  if awk "BEGIN{exit !($DPM - $PRS > 1000)}"; then
    ok "toggling brain changes stranded watts by >1000 W (dpm↔prs)"
  else
    bad "toggle delta too small"
  fi
fi

# DC5 — k8s CRDs reconcile on a real cluster (kind/k3s).
say "DC5: k8s operator reconcile"
if [ "${SKIP_K8S:-0}" = "1" ]; then
  gate "skipped (SKIP_K8S=1)"
elif command -v kubectl >/dev/null 2>&1 && kubectl get crd powerdomains.opendps.io >/dev/null 2>&1; then
  kubectl apply -f deploy/k8s/examples/demo-powerdomain.yaml -n opendps
  kubectl apply -f deploy/k8s/examples/demo-powerpolicy.yaml -n opendps
  # Wait for the operator to write status rather than a fixed sleep.
  for _ in $(seq 1 15); do
    PHASE=$(kubectl get powerdomain demo -n opendps -o jsonpath='{.status.phase}' 2>/dev/null || true)
    [ "$PHASE" = "Active" ] && break
    sleep 1
  done
  if [ "$PHASE" = "Active" ]; then ok "PowerDomain reconciled to phase=Active"; else bad "PowerDomain phase=$PHASE"; fi
  # PowerPolicy reconcile must have written params.json into the domain ConfigMap.
  if kubectl get configmap opendps-topology-demo -n opendps \
       -o jsonpath='{.data.params\.json}' 2>/dev/null | grep -q cap_raise_rate_w_per_tick; then
    ok "PowerPolicy params propagated to domain ConfigMap"
  else
    bad "params.json missing from domain ConfigMap"
  fi
else
  gate "no cluster/CRDs (apply deploy/k8s/crds + operator first)"
fi

# DC6 — CVXPY brain solves optimally.
say "DC6: CVXPY brain optimal solve"
CV_LOG="$(mktemp)"
"${CTL[@]}" --sim --brain cvxpy --config "$CONFIG" --metrics-port 19422 --interval 0.5 >"$CV_LOG" 2>&1 &
CV_PID=$!; sleep 4; kill "$CV_PID" 2>/dev/null || true; wait "$CV_PID" 2>/dev/null || true
if grep -q "cvxpy:optimal" "$CV_LOG"; then ok "CVXPY reports optimal solve"; else bad "no cvxpy:optimal in solver log"; fi

# DC4 — real GPU failsafe latency (hardware-gated).
say "DC4: real-GPU failsafe latency"
gate "requires a cap-capable GPU node (A10/B300/GB200) — run scripts/hw_failsafe.sh there"

# DC7 — time budget.
say "DC7: reproducible within 15 minutes"
ELAPSED=$(( $(ts) - START ))
if [ "$ELAPSED" -lt 900 ]; then ok "completed in ${ELAPSED}s (<900s)"; else bad "took ${ELAPSED}s"; fi

say "SUMMARY"
printf "  %d passed, %d failed (DC4 gated on hardware)\n" "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
