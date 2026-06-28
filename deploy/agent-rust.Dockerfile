FROM rust:1.86-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y pkg-config && rm -rf /var/lib/apt/lists/*
COPY Cargo.toml ./
COPY crates ./crates
RUN cargo build --release --package opendps-agent

FROM debian:bookworm-slim
COPY --from=builder /build/target/release/opendps-agent /usr/local/bin/opendps-agent
CMD ["opendps-agent", \
     "--sim", \
     "--sim-gpus", "10", \
     "--sim-draw-w", "700", \
     "--failsafe-threshold-w", "950", \
     "--failsafe-cap-w", "800", \
     "--failsafe-poll-us", "500", \
     "--metrics-port", "9403"]
