"""Does not use transformers at all: every name below belongs to some other code."""
import os

from mylib import Trainer, Tokenizer

tok = Tokenizer()
enc = tok.encode_plus("hello")
os.environ["TRANSFORMERS_CACHE"] = "/data"
trainer = Trainer(tokenizer=tok, torch_dtype="float16", no_cuda=True)
