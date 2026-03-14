FROM python:3.11-slim

WORKDIR /app

# Install system deps: libgomp1 for LightGBM, coinor-cbc for PuLP MILP optimizer
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    coinor-cbc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Railway injects PORT at runtime; default 8080 for local dev
ENV PORT=8080

EXPOSE 8080

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT}"]
