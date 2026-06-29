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
HOST="${OPENDPS_HOST:-127.0.0.1}"   # scrape target; override for a remote stack
PROM_PORT="${OPENDPS_PROM_PORT:-9090}"
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
  val=$(curl -s "$HOST:$port/metrics" 2>/dev/null \
        | awk '/^opendps_idle_stranded_watts\{/ {print $2}')
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  trap - INT TERM EXIT
  # Emit only a numeric value; empty/non-numeric becomes NaN (caught by caller).
  if [[ "$val" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then echo "$val"; else echo "NaN"; fi
}

is_num() { [[ "$1" =~ ^[0-9]+(\.[0-9]+)?$ ]]; }

# Sum opendps_gpu_power_cap_watts over a set of GPU indices (for N13/DC8).
# Args: <metrics-text> <gpu-index>...   Matches the exact gpu="N"} label suffix
# so gpu="6" never collides with gpu="60".
tenant_cap_sum() {
  local metrics=$1; shift
  local total=0 g v
  for g in "$@"; do
    v=$(printf '%s\n' "$metrics" \
        | awk -v tag="gpu=\"$g\"}" '/^opendps_gpu_power_cap_watts\{/ && index($0, tag) {print $2}')
    is_num "$v" && total=$(awk "BEGIN{print $total + $v}")
  done
  echo "$total"
}

# DC1 — single-workstation stack comes up.
say "DC1: sim stack reachable"
if curl -sf "$HOST:$PROM_PORT/-/healthy" >/dev/null 2>&1; then
  ok "Prometheus healthy on :$PROM_PORT"
else
  gate "Prometheus not up (run: docker compose -f deploy/compose.yml up -d)"
fi

# DC2 — an oversubscribed domain (budget < sum of GPU maxima): DPM strands the
# idle headroom it statically allocates; PRS reclaims it. Scenario is in $CONFIG.
say "DC2: stranded-watts reclaim (oversubscribed domain)"
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
CV_PID=$!
trap 'kill "$CV_PID" 2>/dev/null || true' INT TERM EXIT
sleep 4
kill "$CV_PID" 2>/dev/null || true; wait "$CV_PID" 2>/dev/null || true
trap - INT TERM EXIT
if grep -q "cvxpy:optimal" "$CV_LOG"; then ok "CVXPY reports optimal solve"; else bad "no cvxpy:optimal in solver log"; fi

# DC8 — N13 per-tenant quota enforcement. teamA gets 60% of the 8000 W domain
# budget (GPUs 0-5), teamB 40% (GPUs 6-9). With every GPU busy the brain pins
# each tenant to its slice: teamA's caps must sum to <=4800 W, teamB's <=3200 W.
say "DC8: per-tenant quota enforcement (N13)"
"${CTL[@]}" --sim --brain quota-prs --config "$CONFIG" \
  --quota-config deploy/quota-demo.json --metrics-port 19423 --interval 0.5 >/dev/null 2>&1 &
Q_PID=$!
trap 'kill "$Q_PID" 2>/dev/null || true' INT TERM EXIT
sleep 4
Q_METRICS=$(curl -s "$HOST:19423/metrics" 2>/dev/null || true)
kill "$Q_PID" 2>/dev/null || true; wait "$Q_PID" 2>/dev/null || true
trap - INT TERM EXIT
CAP_A=$(tenant_cap_sum "$Q_METRICS" 0 1 2 3 4 5)
CAP_B=$(tenant_cap_sum "$Q_METRICS" 6 7 8 9)
if is_num "$CAP_A" && is_num "$CAP_B" && awk "BEGIN{exit !($CAP_A>0 && $CAP_B>0)}"; then
  printf "  teamA caps = %.0f W (60%% slice = 4800)\n  teamB caps = %.0f W (40%% slice = 3200)\n" "$CAP_A" "$CAP_B"
  if awk "BEGIN{exit !($CAP_A <= 4800 + 1)}"; then ok "teamA held within its 60% slice (<=4800 W)"; else bad "teamA exceeded slice: $CAP_A W"; fi
  if awk "BEGIN{exit !($CAP_B <= 3200 + 1)}"; then ok "teamB held within its 40% slice (<=3200 W)"; else bad "teamB exceeded slice: $CAP_B W"; fi
else
  bad "quota-prs exported no per-GPU cap metrics (A=$CAP_A B=$CAP_B)"
fi

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
