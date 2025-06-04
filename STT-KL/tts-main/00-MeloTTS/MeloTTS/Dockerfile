FROM python:3.9-slim
WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y \
    build-essential libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first (cache optimization)


# Copy only requirements.txt initially to leverage Docker cache
WORKDIR /workspace
COPY requirements.txt /workspace/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m unidic download

RUN pip install gunicorn==20.1.0
RUN pip install python-mecab-ko

COPY start_server.sh /workspace/
EXPOSE 9880
RUN chmod +x /workspace/start_server.sh
ENTRYPOINT ["/workspace/start_server.sh"]