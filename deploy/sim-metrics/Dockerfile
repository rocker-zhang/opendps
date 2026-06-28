FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir "prometheus-client>=0.20"
COPY sim_exporter.py .
CMD ["python", "sim_exporter.py"]
