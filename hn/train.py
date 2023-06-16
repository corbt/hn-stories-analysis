from dotenv import load_dotenv
from datasets import load_from_disk 
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
import wandb
import os

print("Starting up")

load_dotenv('/workspace/.env')

wandb.login(key=os.getenv("WANDB_API_KEY"))

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
  "microsoft/deberta-v3-base",
  num_labels=1
)
model.to('cuda')

print("Loading dataset...")
dataset = load_from_disk('/workspace/data/hn/stories-dataset')

print(dataset)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base', use_fast=True)

args = TrainingArguments(
  evaluation_strategy = "steps",
  save_strategy="steps",
  eval_steps=3000,
  learning_rate=1e-5,
  per_device_train_batch_size=16,
  per_device_eval_batch_size=32,
  num_train_epochs=1,
  report_to="wandb",
  weight_decay=1e-6,
  output_dir="/workspace/models/hn/frontpage",
  save_total_limit=4,
  warmup_steps=1000,
  # metric_for_best_model="rmse",
  # group_by_length=True,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
  model=model,
  args=args,
  train_dataset=dataset["train"],
  eval_dataset=dataset["test"],
  tokenizer=tokenizer,
  data_collator=data_collator,
)

wandb.init(project="hn-front-page", job_type='train')

# Test on the eval dataset to start
print("Testing...")
trainer.evaluate()

print("Training...")
trainer.train()