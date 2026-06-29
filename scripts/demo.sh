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

# Poll a /metrics endpoint until per-GPU cap lines appear (or timeout). Echoes
# the metrics body. Avoids a fixed sleep that flakes when the controller is slow
# to publish on a loaded CI runner.
wait_for_caps() {
  local url=$1 deadline=$((SECONDS + ${2:-15})) body=""
  while [ "$SECONDS" -lt "$deadline" ]; do
    body=$(curl -s "$url" 2>/dev/null || true)
    if printf '%s\n' "$body" | grep -q '^opendps_gpu_power_cap_watts{'; then
      printf '%s' "$body"; return 0
    fi
    sleep 0.5
  done
  printf '%s' "$body"  # last body (possibly empty) — caller validates
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

# DC8 — N13 per-tenant quota enforcement. tenant-a gets 60% of the 8000 W domain
# budget (GPUs 0-5), tenant-b 40% (GPUs 6-9). In the oversubscribed scenario the
# brain pins each tenant to its slice: tenant-a's caps sum to <=4800 W (busy),
# tenant-b's to <=3200 W (idle, reclaimed below its ceiling).
say "DC8: per-tenant quota enforcement (N13)"
"${CTL[@]}" --sim --brain quota-prs --config "$CONFIG" \
  --quota-config deploy/quota-demo.json --metrics-port 19423 --interval 0.5 >/dev/null 2>&1 &
Q_PID=$!
trap 'kill "$Q_PID" 2>/dev/null || true' INT TERM EXIT
Q_METRICS=$(wait_for_caps "$HOST:19423/metrics" 15)
kill "$Q_PID" 2>/dev/null || true; wait "$Q_PID" 2>/dev/null || true
trap - INT TERM EXIT
CAP_A=$(tenant_cap_sum "$Q_METRICS" 0 1 2 3 4 5)
CAP_B=$(tenant_cap_sum "$Q_METRICS" 6 7 8 9)
if is_num "$CAP_A" && is_num "$CAP_B" && awk "BEGIN{exit !($CAP_A>0 && $CAP_B>0)}"; then
  printf "  tenant-a caps = %.0f W (60%% slice = 4800)\n  tenant-b caps = %.0f W (40%% slice = 3200)\n" "$CAP_A" "$CAP_B"
  if awk "BEGIN{exit !($CAP_A <= 4800 + 1)}"; then ok "tenant-a held within its 60% slice (<=4800 W)"; else bad "tenant-a exceeded slice: $CAP_A W"; fi
  if awk "BEGIN{exit !($CAP_B <= 3200 + 1)}"; then ok "tenant-b held within its 40% slice (<=3200 W)"; else bad "tenant-b exceeded slice: $CAP_B W"; fi
else
  bad "quota-prs exported no per-GPU cap metrics (A=$CAP_A B=$CAP_B)"
fi

# DC9 — N12 job-aware priority boost. GPUs 0,1 carry the same load as 2-5 but
# are marked busy (active job), so --priority-boost lifts their caps above the
# equally-loaded no-job GPUs. A tight budget (topology-jobdemo) makes the boost
# bind instead of everyone sitting at hardware max.
say "DC9: job-aware priority boost (N12)"
"${CTL[@]}" --sim --brain job-prs --config deploy/topology-jobdemo.json \
  --busy-gpus 0,1 --priority-boost 0.30 --metrics-port 19424 --interval 0.4 >/dev/null 2>&1 &
J_PID=$!
trap 'kill "$J_PID" 2>/dev/null || true' INT TERM EXIT
J_METRICS=$(wait_for_caps "$HOST:19424/metrics" 15)
kill "$J_PID" 2>/dev/null || true; wait "$J_PID" 2>/dev/null || true
trap - INT TERM EXIT
BOOSTED=$(tenant_cap_sum "$J_METRICS" 0 1)      # 2 busy/boosted GPUs
PLAIN=$(tenant_cap_sum "$J_METRICS" 2 3 4 5)    # 4 equally-loaded GPUs, no job
if is_num "$BOOSTED" && is_num "$PLAIN" && awk "BEGIN{exit !($BOOSTED>0 && $PLAIN>0)}"; then
  printf "  busy(job) GPU avg cap = %.0f W; no-job GPU avg cap = %.0f W\n" \
    "$(awk "BEGIN{print $BOOSTED/2}")" "$(awk "BEGIN{print $PLAIN/4}")"
  # Compare raw float sums (boosted/2 vs plain/4) to avoid printf rounding at the boundary.
  if awk "BEGIN{exit !($BOOSTED/2 > ($PLAIN/4) * 1.1)}"; then ok "job GPUs boosted above equally-loaded GPUs (>10%)"; else bad "no boost: busy_sum=$BOOSTED plain_sum=$PLAIN"; fi
else
  bad "job-prs exported no per-GPU cap metrics (busy=$BOOSTED plain=$PLAIN)"
fi

# DC10 — N14 multi-node cluster coordination. One busy node + two idle nodes
# share a cluster budget; the coordinator gives the busy node more, the idle
# nodes their floor, and never oversubscribes (Σ budgets <= cluster budget).
say "DC10: multi-node cluster coordination (N14)"
N14_OUT=$(python -m opendps.controller.cluster_coordinator --sim \
  --cluster-budget-w 12000 --nodes node0=8000,node1=500,node2=500 2>/dev/null || true)
node_budget() { printf '%s\n' "$N14_OUT" | awk -v tag="node=\"$1\"}" '/^opendps_cluster_node_budget_w\{/ && index($0,tag){print $2}'; }
B0=$(node_budget node0); B1=$(node_budget node1); TOT=$(printf '%s\n' "$N14_OUT" | sed -n 's/.*total_allocated_w=//p')
if is_num "$B0" && is_num "$B1" && is_num "$TOT"; then
  # is_num rejects scientific notation; fine here since Python prints plain
  # decimals for watt-scale floats (only >~1e15 would format as 1e+15).
  printf "  busy node0 = %.0f W; node1 = %.0f W; total = %.0f W (budget 12000)\n" "$B0" "$B1" "$TOT"
  if awk "BEGIN{exit !($B0 > $B1)}"; then ok "busy node gets a larger cluster budget share"; else bad "busy node not prioritised: node0=$B0 node1=$B1"; fi
  # Hard invariant: never oversubscribe the cluster power budget. The algorithm
  # guarantees Σ==budget exactly bar float epsilon, so the slack is tiny (0.01 W).
  if awk "BEGIN{exit !($TOT <= 12000 + 0.01)}"; then ok "cluster budget not oversubscribed (Σ<=12000 W)"; else bad "oversubscribed: total=$TOT"; fi
else
  bad "coordinator produced no node budgets (B0=$B0 B1=$B1 TOT=$TOT)"
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
