"""Fine-tune a small causal language model, written the Transformers v4 way.

Try it:  transformers-v5-migrate examples/v4_project
The script is only ever analysed, never run: no model is downloaded.
"""
import os

import torch
import transformers as tf
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer as HFTrainer
from transformers import TrainingArguments, pipeline

os.environ["TRANSFORMERS_CACHE"] = "/data/hf-cache"  # v4 cache location

tok = AutoTokenizer.from_pretrained("gpt2", use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    torch_dtype=torch.float16,  # keep this comment
    load_in_8bit=True,
)
reference = AutoModelForCausalLM.from_pretrained("gpt2")  # dtype not given


def tokenize(batch):
    return tok.batch_encode_plus(batch["text"], truncation=True, padding="max_length")


args = TrainingArguments(
    output_dir="out",
    per_gpu_train_batch_size=8,
    no_cuda=True,
    warmup_ratio=0.1,
    group_by_length=True,
    overwrite_output_dir=True,
    include_inputs_for_metrics=True,
)

trainer = HFTrainer(model=model, args=args, tokenizer=tok)
trainer.train(model_path="out/checkpoint-100")

summarizer = tf.pipeline("summarization", model="t5-small")
generator = pipeline("text-generation", model="gpt2")

prompt = tok.apply_chat_template([{"role": "user", "content": "Hello"}], add_generation_prompt=True)
texts = tok.batch_decode(model.generate(**tok("Hello", return_tensors="pt")))
