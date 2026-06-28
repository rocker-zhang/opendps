# opendps

[![CI](https://github.com/rocker-zhang/opendps/actions/workflows/ci.yml/badge.svg)](https://github.com/rocker-zhang/opendps/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)

Open-source reimplementation of a datacenter GPU dynamic power management stack.

## What is this?

opendps implements the telemetry -> PDN modeling -> closed-loop GPU power capping pipeline found in production datacenter power management systems, as a vendorable open-source alternative.

**Key capabilities:**
- **PRS brain** -- EWMA-based idle reclaim: reallocates stranded watts from idle GPUs to hot ones in real time
- **CVXPY optimizer brain** -- LP-based optimal allocation (minimize wasted headroom subject to budget)
- **Rust failsafe** -- sub-millisecond cap enforcement, brain-independent, SCHED_FIFO
- **PyO3 sim backend** -- 1000-GPU simulation runnable from Python brain loop
- **DCGM integration** -- field 160 (power mgmt limit) + field 155 (power draw) via dcgm-exporter
- **Redfish skeleton** -- chassis/NVSwitch power via BMC Redfish API

**Demo result (Phase 1):** 10 GPUs in 8-GPU power budget. PRS brain reclaims >86% of stranded watts. "Stranded watts -> 0" live in Grafana.

**Phase 2:** Rust hot-path P50 latency 63us vs Python 20-50ms.

## Architecture

```
dcgm-exporter -> Prometheus -> Grafana (telemetry visualization)
                    |
              PromClient (opendps)
                    |
              DomainState (per-domain GPU snapshot)
                    |
         PRSBrain / CVXPYBrain / DPMBrain
                    |
            BrainDecision (per-GPU caps)
                    |
         NvmlActuator (Python) / opendps-agent (Rust)
                    |
           NVML nvmlDeviceSetPowerManagementLimit()
```

## Quick start

```bash
# Python brain + sim (no GPU required)
pip install -e ".[dev]"
opendps-controller --sim --config examples/topology-10gpu.json --brain prs --metrics-port 9402

# CVXPY optimizer brain (LP-based optimal allocation)
pip install -e ".[dev,cvxpy]"
opendps-controller --sim --config examples/topology-10gpu.json --brain cvxpy --metrics-port 9402

# Full stack (requires NVIDIA GPU + Docker)
cd deploy && docker compose up

# Rust agent (requires NVML)
cargo build --release -p opendps-agent
./target/release/opendps-agent --nvml --metrics-port 9403
```

## Components

| Component | Language | Description |
|---|---|---|
| `src/opendps/brain/` | Python | DPM, PRS, CVXPY, QuotaAwarePRS brains |
| `src/opendps/controller/` | Python | Standalone control loop |
| `src/opendps/pdn/` | Python | PDN topology model |
| `src/opendps/telemetry/` | Python | Prometheus client, metrics, Redfish |
| `crates/opendps-agent/` | Rust | NVML cap enforcement + failsafe |
| `crates/opendps-sim/` | Rust | PyO3 sim backend |
| `deploy/` | YAML | Docker Compose, Prometheus, Grafana |

## Milestones

| Phase | Status |
|---|---|
| Phase 1 -- PRS brain, oversubscription reclaim demo | Done |
| Phase 2 -- Rust hot-path: failsafe <1ms, PyO3 sim, bench | Done |
| Phase 3 -- Chassis power (IPMI/Redfish), CVXPY brain, job-awareness, quota, multi-node coordinator | Done |

See [ROADMAP.md](docs/ROADMAP.md) for full milestone details.

## Hardware support

| GPU | Telemetry | Power-cap | Notes |
|---|---|---|---|
| GB10 / DGX Spark | Yes | No | Local dev + sim |
| A10 / A100 | Yes | Yes | NVML `-pl` path |
| B300 SXM6 AC | Yes | Yes | Up to ~1100W |
| GB200 | Yes | Yes | Up to ~1200W |

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on pull requests, commit style, and the milestone review process.

## License

Apache 2.0 -- see [LICENSE](LICENSE).
