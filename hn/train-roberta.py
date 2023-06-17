from dotenv import load_dotenv
from datasets import load_from_disk 
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
import os
import wandb

print("Starting up")

load_dotenv('/workspace/.env')

# If the WANDB_NAME environment variable is set, report to wandb
report_to = "none"

if os.getenv('WANDB_NAME'):
  report_to = "wandb"
  wandb.init(project="hn-front-page", job_type='train')

print(f"Reporting to {report_to}")

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
  "xlm-roberta-large",
  num_labels=1
)
model.to('cuda')

print("Loading dataset...")
dataset = load_from_disk('/workspace/data/hn/stories-roberta-dataset')

print(dataset)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large', use_fast=True)

args = TrainingArguments(
  evaluation_strategy = "steps",
  save_strategy="steps",
  eval_steps=5000,
  learning_rate=1e-5,
  per_device_train_batch_size=8,
  per_device_eval_batch_size=32,
  num_train_epochs=3,
  report_to=report_to,
  weight_decay=1e-6,
  output_dir="/workspace/models/hn/frontpage",
  save_total_limit=1,
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

# Test on the eval dataset to start
print("Testing...")
trainer.evaluate()

print("Training...")
trainer.train()