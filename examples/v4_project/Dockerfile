FROM python:3.11-slim

ENV TRANSFORMERS_CACHE=/models/cache
ENV HF_HOME=/models/home

WORKDIR /app
COPY constraints.txt .
RUN pip install -r constraints.txt
COPY . .

RUN transformers-cli download gpt2
CMD ["python", "train.py"]
