# Installation

## Python package (recommended)

```bash
# Basic install (brain + controller, no GPU required)
pip install opendps

# With CVXPY optimizer brain
pip install "opendps[cvxpy]"

# With simulation backend (for testing without GPUs)
pip install "opendps[sim]"

# Full install for development
pip install "opendps[dev,sim,cvxpy]"
```

## Quick demo (no GPU required)

```bash
# Sim mode: 10 GPUs, 8000W budget, CVXPY brain
opendps-controller --sim --brain cvxpy \
  --config deploy/topology-demo.json \
  --metrics-port 9402 --interval 3
```

## Full stack (GPU + Docker)

Requires: Docker, NVIDIA GPU with DCGM support

```bash
cd deploy
docker compose up -d
# Open http://localhost:3000 for Grafana (admin/admin)
# Open http://localhost:9090 for Prometheus
```

## Rust agent (GPU required)

```bash
# Build with NVML support (requires libnvidia-ml.so)
cargo build --release -p opendps-agent --features nvml

# Run on a GPU node
./target/release/opendps-agent --nvml --metrics-port 9403
```

## Hardware requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | Any NVIDIA GPU with DCGM | B300, A100, H100 |
| CPU | x86_64 or aarch64 | — |
| RAM | 512MB | 2GB |
| OS | Linux (kernel 5.10+) | Ubuntu 22.04+ |

## Environment variables

| Variable | Description | Default |
|---|---|---|
| `OPENDPS_METRICS_PORT` | Prometheus metrics port | 9402 |
| `OPENDPS_INTERVAL` | Control loop interval (s) | 5.0 |
| `OPENDPS_BRAIN` | Brain algorithm | prs |
