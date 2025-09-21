# brain-tumor-detection

# With Docker

# 1) Build image
docker build -t tumor-app:latest .

# 2) Run 
docker run --rm -p 8502:8501 \
  -e ROBOFLOW_API_KEY="u_ip_keyʼ \
  tumor-app:latest

# With Streamlit only 

streamlit run app.py \
  --server.enableXsrfProtection=false \
  --server.enableCORS=false