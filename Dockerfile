FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    tcpdump \
    iproute2 \
    libpcap-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]
