FROM python:3.13-slim

WORKDIR /app

# System deps kept minimal; add build-essential only if a future dependency needs compiling.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# AWS App Runner / ECS / Elastic Beanstalk (Docker platform) inject PORT.
# Default to 8080 for local `docker run` testing.
ENV PORT=8080
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",8080)}/_stcore/health')" || exit 1

CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true"]
