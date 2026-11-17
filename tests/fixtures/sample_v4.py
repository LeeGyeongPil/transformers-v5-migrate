"""A v4 style training script: every automatic fix once, plus look-alikes that must stay untouched."""
import torch
import trl
import transformers as tf
from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoTokenizer
from transformers import Seq2SeqTrainer
from transformers import Trainer as HFTrainer
from transformers import TrainingArguments


class MyTrainer(HFTrainer):  # a Trainer subclass defined in this file
    pass


class MyThing:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer


tok = AutoTokenizer.from_pretrained("gpt2", use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    torch_dtype=torch.float16,  # keep this comment
)
vision = AutoModelForVision2Seq.from_pretrained("llava-hf/llava-1.5-7b-hf", dtype=torch.float16)
enc = tok.encode_plus("hello", return_tensors="pt")
batch = tok.batch_encode_plus(["a", "b"], padding=True)

args = TrainingArguments(
    output_dir="out",
    per_gpu_train_batch_size=8,
    per_gpu_eval_batch_size=16,
    no_cuda=True,
    push_to_hub_token="hf_xxx",
    warmup_ratio=0.1,
    include_inputs_for_metrics=True,
)
quiet = TrainingArguments(output_dir="out2", include_inputs_for_metrics=False)

trainer = HFTrainer(model=model, args=args, tokenizer=tok)
trainer2 = tf.Seq2SeqTrainer(model=model, tokenizer=tok)
trainer3 = MyTrainer(model=model, args=args, tokenizer=tok)
trainer4 = Seq2SeqTrainer(model=model, args=args, tokenizer=tok)
trainer.train(model_path="out/checkpoint-100")

other = MyThing(tokenizer=tok)  # not a Trainer: must stay unchanged
sft = trl.SFTTrainer(model=model, tokenizer=tok)  # TRL is a different library: untouched
text = "tokenizer=tok, torch_dtype=x"  # strings are not code
