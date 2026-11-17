"""Looks scary, but nothing here is a v4 problem.

TRANSFORMERS_CACHE is mentioned only in this docstring, and so is encode_plus().
"""
import os

import transformers as tf
from transformers import AutoModelForCausalLM, AutoTokenizer, WhisperFeatureExtractor


class MyThing:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer


# comment: tok.encode_plus("x") and os.environ["TRANSFORMERS_CACHE"] are only text here
tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2", dtype="auto")
thing = MyThing(tokenizer=tok)
audio = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
generator = tf.pipeline("text-generation", model="gpt2")
label = "summarization"
home = os.environ["HOME"]
decoded = tok.decode([1, 2, 3])
model.save_pretrained("out", max_shard_size="2GB")
tok.save_pretrained("out")
prompt = tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False)
settings = {"learning_rate": 1e-4, "tokenizer": "gpt2"}
