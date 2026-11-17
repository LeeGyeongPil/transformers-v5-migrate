FROM python:3.11-slim

ENV TRANSFORMERS_CACHE=/models/cache
ENV HF_HOME=/models/home

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

RUN transformers-cli download gpt2
CMD ["python", "train.py"]
