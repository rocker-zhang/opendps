FROM python:3.12-slim
WORKDIR /app
# Copy only what's needed to install the package
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[sim,prometheus]"
# Default topology file is provided at compose time via mount or baked in
COPY deploy/topology-demo.json /app/topology-demo.json
CMD ["opendps-controller", \
     "--sim", "--brain", "prs", \
     "--config", "/app/topology-demo.json", \
     "--metrics-port", "9402", \
     "--interval", "3"]
