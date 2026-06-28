# opendps — Architecture

## Design principle: thin self-built core, integrate OSS for the rest

opendps reimplements the *ideas* of a datacenter GPU dynamic power-management
stack (reference: nvidia-smi power control, WPPS, DPS/DPM/PRS). We do **not**
reinvent the parts that already exist as mature open source. We build only the
moat and wire it through a Kubernetes **operator** into the existing ecosystem,
shipping everything as **one integrated project** (compose/Helm brings the whole
stack up).

### Reused open-source components (integrated, not rebuilt)

| Concern | Component | Why |
|---|---|---|
| GPU telemetry | **dcgm-exporter** | native, exports every DCGM field as Prometheus metrics |
| Metrics store / query | **Prometheus** | time series + PromQL, the brain reads from here |
| Visualization | **Grafana** | dashboards as JSON config, no custom web UI |
| Scheduling / node mgmt | **Kubernetes** + **NVIDIA GPU Operator** | job placement, device plugin, node lifecycle |
| Cap-push mechanism | **DCGM / NVML / dcgmi** | the driver-level knob (`nvmlDeviceSetPowerManagementLimit`) |

### Self-built core (the moat)

| Component | Role |
|---|---|
| **opendps-operator** | k8s controller. Reconciles `PowerDomain` / `PowerPolicy` CRDs. Reads telemetry from Prometheus (PromQL), runs the control brain, computes per-GPU caps, dispatches to node agents. The integration point that wires OSS telemetry → brain → enforcement. |
| **control brain** (library inside the operator) | `collect → predict (EWMA/percentile) → solve per power-domain → allocate (proportional/priority) → emit caps`. DPM (static per-node cap) and PRS (dynamic cross-GPU headroom reallocation, oversubscription reclaim). |
| **opendps-agent** | node DaemonSet. Applies the per-GPU cap via NVML/dcgmi. Hosts the **failsafe fast-loop**: brain-independent, dual-time-scale, can only *lower* caps under fault. |
| **sim backend** (digital twin) | GPU power model + PDN topology/capacity. Lets the same operator drive a simulated fleet larger than the physical lab, for single-workstation reproducible demos. The agent's enforcement target is pluggable: real node (NVML/dcgmi) or sim. |

### Data flow

```
 [OSS] dcgm-exporter (DaemonSet, per node)
            │ scrape
            ▼
 [OSS] Prometheus ──────────────┐ PromQL (telemetry in)
                                ▼
                  [SELF] opendps-operator  ◄──── PowerDomain / PowerPolicy CRDs
                     control brain: predict → solve → allocate
                                │ desired per-GPU caps
                                ▼
                  [SELF] opendps-agent (DaemonSet)        ── or ──►  [SELF] sim backend
                     NVML/dcgmi setPowerManagementLimit               digital-twin fleet
                     + failsafe fast-loop (cap-lower-only)
                                │
                                ▼
                          GPU hardware power
                                │ (loop closes via dcgm-exporter)
 [OSS] Grafana ◄── Prometheus   ┘  visualize domains / caps / headroom / reclaim
```

### Demo topologies

1. **Single-workstation (reproducible by anyone):** operator + sim backend, no
   GPU required for the control-loop story; Grafana shows oversubscription
   reclaim and peak shaving on the simulated fleet.
2. **Real hardware (B300 / GB200):** operator + real opendps-agent enforcing
   caps via dcgm/NVML; dcgm-exporter feeds real telemetry; same Grafana.

GB10/DGX Spark (local dev board) cannot power-cap, so it serves local development
and the telemetry/dashboard path. Closed-loop enforcement is validated on any
GPU with DCGM + power-cap support (e.g. A10, B300, GB200).

_Last updated: 2026-06-27_
