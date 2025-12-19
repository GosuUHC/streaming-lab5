FROM python:3.10-slim

WORKDIR /app

ENV ONNXRUNTIME_DISABLE_EXECUTABLE_STACK_CHECK=1

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    find /usr/local/lib/python3.10/site-packages/onnxruntime/capi -name "*.so" -exec patchelf --clear-execstack {} \; 2>/dev/null || true

COPY hospital_readmissions_30k.csv .
COPY train_model.py .
COPY app.py .
COPY entrypoint.sh .

RUN python train_model.py && \
    chmod +x entrypoint.sh

EXPOSE 8000

CMD ["./entrypoint.sh"]