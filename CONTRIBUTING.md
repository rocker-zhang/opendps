# Contributing to opendps

## Getting started

Clone the repo and set up a project-local venv (never install into system Python):

```bash
git clone https://github.com/OWNER/opendps
cd opendps
./scripts/setup-venv.sh dev
source .venv/bin/activate
```

## Running tests

```bash
pytest tests/ -x
```

Most tests run without a GPU. Tests that require NVML or DCGM are skipped automatically when the hardware is absent.

## Building the Rust components

```bash
# Agent binary (requires NVML headers on Linux)
cargo build -p opendps-agent

# Sim backend (PyO3, no hardware required)
cargo build -p opendps-sim
```

Release builds:

```bash
cargo build --release -p opendps-agent
```

## Running the sim demo

```bash
# Headless: Python brain + PyO3 sim, no GPU required
opendps-controller --sim --config deploy/topology-demo.json --brain prs --metrics-port 9402

# Full compose stack (Docker required)
docker compose -f deploy/compose.yml up
```

Open Grafana at http://localhost:3000.

## Pull request process

- One milestone per PR. Keep PRs focused; large multi-milestone PRs will be split.
- Tests must pass (`pytest tests/ -x` and `cargo test` green).
- New behavior needs a test. If a test is not feasible (hardware-only path), add a note explaining why.
- Update `docs/ROADMAP.md` if the PR completes or advances a milestone.
- Do not include any internal infrastructure information in PRs, issues, or commit messages: no internal hostnames, IP ranges, VPN configs, jump proxies, SSH key paths, or employer identifiers.

## Code style

- Python: `ruff check` and `ruff format` (config in `pyproject.toml`).
- Rust: `cargo fmt` and `cargo clippy -- -D warnings`.

CI runs both automatically on every PR.
