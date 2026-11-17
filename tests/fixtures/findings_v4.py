"""Report-only cases. A trailing comment of the form "expect: IDS" lists the findings expected on that line."""
import os

import torch
import transformers as tf
from transformers import (
    AutoModelForCausalLM,
    AutoModelWithLMHead,  # expect: TV5M04
    CLIPFeatureExtractor,  # expect: TV5M11
    TFAutoModel,  # expect: TV5M03
    TrainingArguments,
)

tok = tf.AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained(  # expect: TV5B01
    "gpt2",
    load_in_8bit=True,  # expect: TV5M01
)
legacy = AutoModelForCausalLM.from_pretrained(  # expect: TV5B01
    "gpt2-tf",
    from_tf=True,  # expect: TV5M02
)
head = AutoModelWithLMHead.from_pretrained("gpt2", dtype=torch.float16)
tf_model = tf.TFAutoModel.from_pretrained("bert-base-uncased")  # expect: TV5M03
extractor = tf.ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")  # expect: TV5M11

summarizer = tf.pipeline("summarization", model="t5-small")  # expect: TV5M05
translator = tf.pipeline(task="translation", model="t5-small")  # expect: TV5M05
generator = tf.pipeline("text-generation", model="gpt2")

ratio = 0.1
flag = True
args = TrainingArguments(
    output_dir="out",
    overwrite_output_dir=True,  # expect: TV5M06
    group_by_length=True,  # expect: TV5M06
    logging_dir="runs",  # expect: TV5M06
    warmup_ratio=ratio,  # expect: TV5M06
    include_inputs_for_metrics=flag,  # expect: TV5M06
)
clash = TrainingArguments(
    output_dir="out",
    warmup_ratio=0.1,  # expect: TV5M06
    warmup_steps=10,
)

batch = tok.prepare_seq2seq_batch(["a"])  # expect: TV5M07
with tok.as_target_tokenizer():  # expect: TV5M07
    labels = tok(["b"])

os.environ["TRANSFORMERS_CACHE"] = "/data/hf"  # expect: TV5M08
fast = os.getenv("HF_HUB_ENABLE_HF_TRANSFER")  # expect: TV5M08
cache = os.environ.get("PYTORCH_PRETRAINED_BERT_CACHE", "/x")  # expect: TV5M08
os.environ.setdefault("HUGGINGFACE_CO_RESOLVE_ENDPOINT", "https://example.org")  # expect: TV5M08

model.save_pretrained("out", safe_serialization=False)  # expect: TV5M09, TV5B04

cfg = {
    "torch_dtype": "float16",  # expect: TV5M12
    "use_auth_token": True,  # expect: TV5M12
    "batch_size": 8,
}
opts = dict(no_cuda=True)  # expect: TV5M12

from_pretrained_dup = AutoModelForCausalLM.from_pretrained("x", torch_dtype=torch.float16, dtype=torch.float32)  # expect: TV5R01


def make_trainer():
    ...


make_trainer().train(model_path="out")  # expect: TV5R04

ids = [1, 2, 3]
text = tok.batch_decode(ids)  # expect: TV5B03
chat = tok.apply_chat_template([{"role": "user", "content": "hi"}])  # expect: TV5B02
plain = tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False)
old_style = tok.apply_chat_template([], return_dict=False)
classifier = tf.AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")  # expect: TV5B01
tok.save_pretrained("out")
