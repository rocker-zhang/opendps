# opendps — Roadmap

Open-source, vendor-neutral reimplementation of a datacenter GPU dynamic
power-management stack (telemetry → PDN modeling → closed-loop capping).

**Final goal (updated 2026-06-28):**  
Achieve functional parity with closed-source NVIDIA DPS across three dimensions:

| Dimension | Status |
|---|---|
| **Phase 1** — PRS brain, oversubscription reclaim demo (sim + B300/GB200 real-hardware) | ✅ Done |
| **Phase 2** — Rust hot-path: sub-ms failsafe *detection*, PyO3 sim backend, bench_failsafe | ✅ Done |
| **Phase 3** — Close telemetry + control gaps vs closed-source DPS (N8–N14) | ✅ Done |

**Phase 3 gap summary (from agent analysis, 2026-06-28):**

| Gap | Severity | Target milestone |
|---|---|---|
| Chassis/node power (IPMI `dcmi power reading`) | Critical — budget off 15-20% | N9 ✅ (RAPL on B300, IPMI config) |
| NVSwitch + CPU overhead in PDN model | Critical | N9 ✅ |
| Redfish client (NVLink Switch chassis, PDU rails) | Important | N10 ✅ (skeleton — needs BMC NIC) |
| CVXPY optimal-allocation brain (vs EWMA heuristic) | Important | N11 ✅ |
| Job-awareness (GPU → job mapping via `nvidia-smi --query-compute-apps`) | Important | N12 ✅ (configurable `--priority-boost`; wired into the demo path as `demo.sh` DC9; [design doc](N12-job-awareness.md)) |
| Production hardening (/healthz, alerting, watchdog, config reload) | Important | N9 ✅ |
| Per-tenant quota enforcement | Nice | N13 ✅ (QuotaAwarePRSBrain, config-driven via `--quota-config`; wired into the demo path as `demo.sh` DC8; [design doc](N13-quota-enforcement.md)) |
| Multi-node cluster coordinator | Nice | N14 ✅ (proportional rebalancer with a hard Σ≤budget invariant; CLI + sim demo `demo.sh` DC10; [design doc](N14-multinode.md)) |

> Validation depth varies by milestone. N0–N7 are exercised end-to-end (sim +
> real GPU node); N12 (job-aware boost), N13 (per-tenant quota) and N14
> (multi-node coordination) are now config/CLI-driven and exercised in the sim
> demo (`demo.sh` DC9 / DC8 / DC10). N10 (Redfish) remains a skeleton not wired
> into the default demo path — it needs a BMC management NIC to exercise live;
> and N14's live multi-process run still needs a Redis-backed cluster (the demo
> uses the in-memory store in one process).

This is a clean-room reimplementation of the ideas behind NVIDIA DPS/DPM/PRS.
The closest commercial equivalent is [Pebble](https://www.gopebble.com) (closed
source). The OSS ecosystem (EAR, GEOPM, Kepler, Zeus) covers HPC / per-job
tuning but not the specific combination we are building. See ARCHITECTURE.md.

---

## Real moat (defensible whitespace vs OSS + Pebble)

Pebble ships a closed-source GPU + k8s product. No OSS project combines:
- k8s-native **PowerDomain / PowerPolicy / JobPowerPolicy CRDs**
- Self-owned **facility PDN topology + capacity model** as solver hard constraints
- **Oversubscription reclaim** across a heterogeneous (non-Blackwell) fleet
- **GPU dual-loop failsafe**: brain-independent, cap-lower-only, dual time-scale

That specific intersection is the project.

---

## Reused open-source (integrated, not rebuilt)

| Concern | Component |
|---|---|
| GPU telemetry (dashboard / history) | dcgm-exporter → Prometheus → Grafana |
| GPU telemetry (control loop, low-latency) | direct DCGM API |
| Cap-push mechanism | NVML / dcgmi (`nvmlDeviceSetPowerManagementLimit`) |
| Scheduling substrate | Kubernetes + NVIDIA GPU Operator |
| Allocation solver math | CVXPY or OR-Tools (borrow GEOPM `power_balancer` approach, BSD-3) |
| CRD scaffolding | kubebuilder / controller-runtime |

---

## Self-built components

| Component | Role |
|---|---|
| **Telemetry source** | Per-GPU draws into the control loop — from dcgm-exporter via Prometheus/PromQL, or directly from the NVML actuator (`--telemetry actuator`) on a bare GPU node |
| **PDN model** | Facility power-distribution topology + capacity as solver constraints |
| **Standalone controller + brain** | Zero-k8s process: telemetry → brain (predict→solve→allocate) → push caps; k8s operator wraps the same library later |
| **Brain v1 (DPM)** | Per-node static cap enforcement |
| **Brain v2 (PRS)** | EWMA predict + CVXPY solver + proportional/priority allocation + oversubscription reclaim |
| **opendps-agent** | Node-level NVML/dcgmi cap enforcement + software failsafe (cap-lower-only fast loop). Runs as process OR DaemonSet |
| **Sim backend** | GPU power model + PDN topology, calibrated from real telemetry; enables fleet-scale single-workstation demo |
| **k8s operator + 3 CRDs** | Production wrapper around the same brain library |
| **Job/policy intake** | Scheduling→brain data contract (job boundary, priority, NVLink topology, step rhythm) |

---

## Target hardware

The project targets any GPU with DCGM + power-cap support. GB10/DGX Spark is the
local dev board (telemetry only; no power-cap). The cap enforcement path is
validated on B300 and GB200 hardware. dcgm-exporter requires the DCGM daemon,
so it cannot run on devices lacking DCGM (e.g. GB10).

The agent supports **process mode** (plain binary, no DaemonSet) so k8s is not
required to validate the cap-enforcement path.

| GPU family | telemetry | power-cap | typical role |
|---|---|---|---|
| GB10 / DGX Spark | yes | **no** | local dev + sim demo |
| A10 / A100 | yes | yes (`-pl`) | NVML cap path validation |
| B300 SXM6 | yes | yes (dcgmi, up to ~1100 W) | real closed-loop demo |
| GB200 | yes | yes (dcgmi, up to ~1200 W) | real closed-loop demo |

---

## Milestones (demo-first vertical-slice ordering)

Ordering rationale: sim before agent, brain loop before operator wrapper,
first complete demo vertical slice at N1 — not at the end.

| # | Name | What gets built | First demoable at |
|---|---|---|---|
| **N0** | Compose stack + PromQL client | `docker compose up` brings up dcgm-exporter + Prometheus + Grafana; thin PromQL client library reads metrics into Python; sim metric source for hosts without DCGM | Grafana shows live GPU power |
| **N1** ⭐ | **First vertical slice** | Direct-DCGM collector + PDN model (simple topology) + sim backend + standalone controller + brain v1 (static DPM) + Grafana panel | **Full loop on sim: telemetry→brain→cap→Grafana** |
| **N2** | PRS brain + oversubscription reclaim | Brain v2: EWMA→CVXPY solver→proportional/priority alloc + oversub reclaim switch; calibrate sim from real telemetry | PRS on/off toggle, stranded-watts counter drops |
| **N3** | Real hardware enforcement | opendps-agent: NVML/dcgmi cap + software failsafe; runs as process OR DaemonSet; validated on cap-capable GPU hardware | Same brain loop with real caps |
| **N4** ✅ | k8s operator + CRDs | kopf operator wrapping same brain library; PowerDomain / PowerPolicy / JobPowerPolicy CRDs | ✅ Reconciles real CRs on a kind cluster (PowerDomain→phase=Active, ConfigMap written, clean delete); RBAC + image-pull + handler-registration bugs fixed |
| **N5** ✅ | Failsafe hardening + training-transient smoothing | PRS cap-raise rate limiter (lowering stays immediate); failsafe/smoothing params carried by PowerPolicy → ConfigMap `params.json` → controller | ✅ Param change propagates to ConfigMap on cluster; rate limiter bounds cap-raise slope (tests) |
| **N6** ✅ | Job/policy intake + data contract | JobPowerPolicy → real matched-pod count + boost registry ConfigMap; JobAwarePRSBrain boosts busy GPUs; sim busy-set for driverless demo | ✅ matchedPods reflects a real matching pod on cluster (not hardcoded 0); busy GPU boosted (tests) |
| **N7** 🎯 ✅ | **FINAL DEMO** | `scripts/demo.sh` acceptance check: sim fleet + brain-agnostic stranded-watts + PRS/DPM toggle + CVXPY optimal + k8s reconcile; `hw_failsafe.sh` + `--telemetry actuator` for the real-GPU loop | ✅ demo.sh green on all sim criteria: DPM 3480 W → PRS 536 W (**85% reclaim**, sim). Real GPU node validated: NVML cap round-trip, Rust failsafe trip (NVML round-trip ~23 ms), closed-loop PRS reclaim 1100 W→~305 W — see docs/hardware-validation.md |

---

## Headline demo scenario (the money shot)

One power domain, **budget = 8-GPU equivalent, 10 GPUs present (25% oversubscribed)**, mixed load.

- Static DPM: every GPU hard-capped to 80%; hot GPUs throttle while idle GPUs strand power → ~10–15% throughput lost
- PRS (N2+): idle budget reallocated to hot GPUs in real time → throughput recovered, **domain under budget 100% of the time**
- **Headline metric**: "PRS recovers >90% of the throughput static capping strands — ~2 extra GPUs of useful work per domain at zero extra power."
- Dashboard: live PRS on/off toggle + "stranded watts" counter falling to ~0.

---

## Process (see AGENTS.md)

Each milestone: dedicated subagent → three-layer review (code / arc / prose) → lab run dual-reviewed → maintainer OK → push.
No push without explicit maintainer approval.

_Last updated: 2026-06-28_

---

## N8 — Chassis power integration (IPMI + DCGM field 160)

**Motivation**: Analysis of closed-source NVIDIA DPS vs opendps reveals a ~15–20% power undercount.  
opendps sees only per-GPU draw (NVML/DCGM). A real datacenter node's total draw includes:

| Component | Approximate power | Visibility before N8 |
|---|---|---|
| GPUs (4× GB200) | 4 × 1200W max | ✅ DCGM field 155 |
| NVSwitches / NVLink | ~200–400W | ❌ not tracked |
| CPU + DRAM + storage | ~300–500W | ❌ not tracked |
| GPU power cap (what brain set) | needed for feedback | ❌ DCGM field 160 missing from default config |

**Deliverables**:
1. **`deploy/dcgm-fields.csv`** — custom field config with field 160 (`POWER_MGMT_LIMIT`) + violation counters + temperature + energy.  
   Mounted into dcgm-exporter container: `./dcgm-fields.csv:/etc/dcgm-exporter/counters.csv:ro`
2. **IPMI exporter** — `prometheus-community/ipmi-exporter` added to `compose.yml`. Scrapes `ipmitool dcmi power reading` for total node chassis draw.
3. **PDN overhead** — `PDNTopology` gains `pdu_overhead_fraction` (default 0.10) + `node_overhead_w` (per-node non-GPU draw from IPMI). Used by PRS brain's budget calculation.
4. **Prometheus targets** — `dcgm-gb200` scrape job targeting live GB200 dcgm-exporter (already added).
5. **Grafana panels** — "GPU cap vs draw" (field 160 vs 155), "chassis vs GPU power" ratio, "power violation throttle rate".

**Done-when**: Grafana shows `dcgm_fi_dev_power_mgmt_limit` populated by dcgm-exporter after a brain cap decision + IPMI chassis power in a separate panel.

---

## Phase 2 — Rust hot path

**Final goal: complete a Phase 2 demo where system-level hot-path components run in Rust.**

Rationale: the failsafe fast-loop, NVML cap enforcement, and direct-DCGM collector are latency-sensitive and run as long-lived daemons on every GPU node — prime candidates for Rust (memory safety, no GIL, zero-cost abstractions). Python retains orchestration and brain logic.

Phase 2 follows the same process as Phase 1: agent team analysis → adversarial review → locked milestone list → per-component isolated subagent build → real hardware validation → demo.

**Candidate components** (to be confirmed by agent analysis — not pre-decided):
- `opendps-agent`: NVML/dcgmi cap enforcement + failsafe fast-loop (highest priority)
- Direct-DCGM collector: tight sampling loop
- PDN model: pure computation, possible Rust library with PyO3 bindings
- Standalone controller core: the control loop tick

Phase 2 planning launches in parallel with Phase 1 N1+ build.

_Last updated: 2026-06-28_

### Phase 2 locked milestones

Synthesized from three adversarial critics (architecture/boundary, prior-art/ecosystem, demo/feasibility).

| # | Milestone | What gets built | Done-when |
|---|---|---|---|
| **P2-M1** | Rust `opendps-agent` — NVML enforcement binary | Standalone Rust binary. `nvml-wrapper ^0.12` + `tokio ^1`. `set_power_management_limit()` validated on cap-capable GPU hardware. Same cap IPC contract as Python agent. | Real cap round-trip via Rust binary |
| **P2-M2** | Failsafe fast-loop in Rust | Brain-independent cap-lower-only state machine. Dedicated `std::thread` + `SCHED_FIFO`. Sub-ms response target. | Injected overload trips in <1ms |
| **P2-M3** | `bench_failsafe` + Grafana `failsafe_latency_microseconds` panel | Benchmark binary: P50/P99 μs. Grafana histogram with "overload injected" / "cap applied" annotations. | Grafana shows <1ms Rust vs 20-50ms Python |
| **P2-M4** | Rust sim backend via PyO3 | `pyo3 ^0.28`, 5-method `Actuator` Protocol as PyO3 class. `allow_threads()` on mutating methods. Concurrent-caller integration test. | 1000-GPU sim runnable from Python brain |
| **P2-M5** ⭐ | **Phase 2 FINAL DEMO** | Rust hot path (agent failsafe + NVML cap) + Python CVXPY brain + Grafana: PRS toggle + failsafe latency panel + stranded-watts counter. Screenshots → Notion. | **Phase 2 demo complete** |
| **P2-M6** *(conditional)* | Direct-DCGM collector in Rust | Only execute if P2-M1 NVML bindings create a natural extension AND profiling shows need | — |

**Rust boundary decisions** (locked, do not revisit without evidence):
- REWRITE: `opendps-agent` (NVML + failsafe fast-loop)
- OPTIONAL: sim backend (PyO3 Actuator boundary)
- SKIP: PDN model, controller + brain (CVXPY dependency), k8s operator (no latency req)

**Deployment path**: aarch64 dev boards: native via rustup. x86_64 target (e.g. datacenter GPU nodes): cross-compile with `cross` tool (`-C target-feature=+crt-static`), `scp` static binary — no runtime deps.

_Phase 2 plan locked: 2026-06-27_

## N9 — Production hardening: IPMI + /healthz + alerting

**Done**: 2026-06-28

**Deliverables**:
- `deploy/ipmi-exporter.yml` + compose service (`profiles: [ipmi]`)
- `/healthz` endpoint in `opendps/telemetry/metrics.py` on `metrics_port + 1`
- `deploy/alerting.yml` — PowerDomainOverBudget, FailsafeTripRateHigh, IdleStrandedWattsHigh
- Grafana panels 10-13: per-GPU draw, cap vs draw, DCGM field 160 stat, failsafe trips
- `PowerDomain.node_overhead_w` + `available_gpu_budget_w` property (the field
  landed here; the control-loop wiring is N18)

## N10 — Redfish chassis power client

**Status**: Skeleton implemented; requires BMC management NIC for live data

**Deliverables**:
- `src/opendps/telemetry/redfish_scraper.py` — `RedfishScraper` + `ChassisPowerReading`
- Parses chassis total, NVSwitch (Oem.Nvidia.NVSwitchPowerWatts), PSU rails
- Integration with PDNTopology `node_overhead_w` = `chassis_total_w - gpu_aggregate_w`

**Done-when**: `chassis_total_w` Prometheus metric populated on a node with BMC NIC access.

## N11 — CVXPY optimizer brain

**Status**: Implemented — LP formulation, GLPK solver, PRS fallback

**Deliverables**:
- `src/opendps/brain/cvxpy_brain.py` — `CVXPYBrain` with LP minimize-wasted-headroom
- `--brain cvxpy` in StandaloneController
- Solve time < 50ms for 10 GPUs
- Graceful PRS fallback if cvxpy unavailable

## N12 — Job-awareness (nvidia-smi process tracking)

**Status**: Implemented — configurable boost (`--priority-boost` / `params.json`),
exercised in the sim demo (`demo.sh` DC9). Design doc:
`docs/N12-job-awareness.md`.

**Deliverables**:
- `src/opendps/agent/job_tracker.py` — `JobTracker` polls nvidia-smi
- `src/opendps/brain/job_aware_prs.py` — configurable priority boost for GPUs
  with active jobs (renormalised to the domain budget)
- `--brain job-prs` + `--priority-boost FRAC` in StandaloneController

**Done-when**: GPU with active CUDA process receives a configurable cap boost vs
an equally-loaded idle-job GPU. ✅ (DC9 asserts the busy GPUs are capped clearly
above equally-loaded no-job GPUs.)

## N13 — Per-tenant quota enforcement

**Status**: Implemented — config-driven (`--quota-config`), exercised in the sim
demo (`demo.sh` DC8). Design doc: `docs/N13-quota-enforcement.md`.

**Key design**:
- `TenantQuota(tenant_id, domain_name, gpu_indices, max_watts_pct)` +
  `QuotaConfig` (JSON via `--quota-config` / `quota.json` next to the topology)
- `QuotaAwarePRSBrain` enforces per-tenant budget slices before intra-tenant PRS
- Two tenants with a 60%/40% split receive proportional GPU budgets

## N14 — Multi-node cluster coordinator

**Status**: Implemented — proportional rebalancer with a hard `Σ≤budget`
invariant, a CLI/sim entry point, and a demo (`demo.sh` DC10). Design doc:
`docs/N14-multinode.md`.

**Deliverables**:
- `src/opendps/controller/cluster_coordinator.py` — proportional budget
  rebalancer (floor-reserve + proportional surplus + ceiling; never
  oversubscribes the cluster budget) + `python -m ...cluster_coordinator` CLI
- `InMemoryStore` for testing/sim; `RedisStore` for production (mock-tested, no live Redis required for CI)
- `ClusterCoordinator.rebalance()` redistributes based on node draw
- `src/opendps/controller/agent_bridge.py` — controller→agent IPC skeleton (newline-delimited JSON over TCP; falls back to no-op if agent unreachable)
- Redis mock tests: `test_redis_store_publish_and_get_all`, `test_redis_store_publish_sets_ttl`
- Agent bridge tests: `test_push_caps_returns_false_when_unreachable`, `test_push_caps_sends_json_messages`

**Full Redis integration**: `pip install ".[redis]"` + running Redis instance required for live use.
**Agent IPC**: `AgentBridge` connects to Rust opendps-agent on `127.0.0.1:9500` (future N4+ work).

---

# Phase 4 — closed-source parity (N15+)

Candidates from the feature-parity analysis (closed-source NVIDIA datacenter GPU
DPM vs opendps). Built in priority order, each sim-validated.

## N18 — Node overhead wired into the GPU budget

**Status**: Implemented.

The N9 model already had `PowerDomain.node_overhead_w` +
`available_gpu_budget_w`, but every brain allocated against the raw `budget_w`,
so CPU/NVSwitch/memory overhead was never reserved. N18 makes
`PDNTopology.domain_budget_w()` return the overhead-adjusted budget, so all
brains (DPM/PRS/CVXPY/quota/job) size GPU caps against the power actually left
for the GPUs. `validate_allocation()` checks GPU caps against
`available_gpu_budget_w` and rolls a domain's `caps + node_overhead_w` into the
PDU total. Default overhead is 0, so existing topologies are unchanged.

**Done-when**: with a non-zero `node_overhead_w`, GPU caps sum to ≤
`budget_w − node_overhead_w`. ✅ (`tests/test_node_overhead_n18.py`)

## N15 — SLA-tiered priority preemption

**Status**: Implemented.

`priorityClass` was captured by the operator but never read back by any brain.
N15 adds `PriorityTieredPRSBrain` (`--brain priority-prs` + `--gpu-priority-tiers`):
under contention the budget surplus is split by `draw × tier_weight`
(low 0.5 / normal 1.0 / high 2.0 / critical 4.0), so higher-tier GPUs keep more
cap while every GPU keeps a floor and `Σcaps ≤ budget`. Design doc:
`docs/N15-priority-preemption.md`.

**Done-when**: under equal load and budget pressure, a higher-tier GPU is capped
above a lower-tier one. ✅ (`demo.sh` DC11; `tests/test_brain_priority_n15.py`)
