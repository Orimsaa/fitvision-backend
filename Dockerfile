# ── Stage 1: Build ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build essentials (needed by some packages like scikit-learn)
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Copy only the slim requirements for deploy
COPY requirements-fly.txt .
RUN pip install --no-cache-dir --user -r requirements-fly.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application files
COPY app/ ./app/
COPY config/ ./config/
COPY src/ ./src/

# Download actual models from Git LFS since Koyeb clone only pulls pointers
RUN mkdir -p data/models && \
    wget -qO data/models/benchpress_form.pkl https://github.com/Orimsaa/fitvision-backend/raw/main/data/models/benchpress_form.pkl && \
    wget -qO data/models/deadlift_form.pkl https://github.com/Orimsaa/fitvision-backend/raw/main/data/models/deadlift_form.pkl && \
    wget -qO data/models/squat_form.pkl https://github.com/Orimsaa/fitvision-backend/raw/main/data/models/squat_form.pkl && \
    wget -qO data/models/squat_form_detailed.pkl https://github.com/Orimsaa/fitvision-backend/raw/main/data/models/squat_form_detailed.pkl && \
    wget -qO data/models/exercise_classifier.pkl https://github.com/Orimsaa/fitvision-backend/raw/main/data/models/exercise_classifier.pkl

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
