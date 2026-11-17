"""A v4 style training script: every automatic fix once, plus look-alikes that must stay untouched."""
import torch
import trl
import transformers as tf
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
from transformers import Seq2SeqTrainer
from transformers import Trainer as HFTrainer
from transformers import TrainingArguments


class MyTrainer(HFTrainer):  # a Trainer subclass defined in this file
    pass


class MyThing:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer


tok = AutoTokenizer.from_pretrained("gpt2", token=True)
model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    dtype=torch.float16,  # keep this comment
)
vision = AutoModelForImageTextToText.from_pretrained("llava-hf/llava-1.5-7b-hf", dtype=torch.float16)
enc = tok("hello", return_tensors="pt")
batch = tok(["a", "b"], padding=True)

args = TrainingArguments(
    output_dir="out",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    use_cpu=True,
    hub_token="hf_xxx",
    warmup_steps=0.1,
    include_for_metrics=["inputs"],
)
quiet = TrainingArguments(output_dir="out2")

trainer = HFTrainer(model=model, args=args, processing_class=tok)
trainer2 = tf.Seq2SeqTrainer(model=model, processing_class=tok)
trainer3 = MyTrainer(model=model, args=args, processing_class=tok)
trainer4 = Seq2SeqTrainer(model=model, args=args, processing_class=tok)
trainer.train(resume_from_checkpoint="out/checkpoint-100")

other = MyThing(tokenizer=tok)  # not a Trainer: must stay unchanged
sft = trl.SFTTrainer(model=model, tokenizer=tok)  # TRL is a different library: untouched
text = "tokenizer=tok, torch_dtype=x"  # strings are not code
