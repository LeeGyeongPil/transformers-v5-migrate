#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_PRETRAINED_BERT_CACHE=/data/hf-cache
python train.py
