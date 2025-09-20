# --- Base image ---
FROM python:3.11-slim

# --- System deps (opencv/video/SSL/ffmpeg helpful) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# --- Workdir ---
WORKDIR /app

# --- Copy requirements early (better layer caching) ---
COPY requirements.txt constraints.txt ./

# Faster, reproducible installs
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt -c constraints.txt

# --- Copy app code ---
COPY . .

# Streamlit server settings: bind to 0.0.0.0, disable CORS/XSRF for local/containers
ENV STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

# Roboflow API key is passed at runtime (never bake secrets)
# ENV ROBOFLOW_API_KEY=""

EXPOSE 8501

# Basic healthcheck (ping streamlit root)
HEALTHCHECK --interval=30s --timeout=3s --retries=3 CMD \
  wget -qO- http://localhost:8501/_stcore/health || exit 1

# --- Entrypoint ---
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
