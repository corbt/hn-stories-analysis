from dotenv import load_dotenv
from datasets import load_from_disk 
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import mean_squared_error
import wandb

print("Starting up")

load_dotenv()

# wandb.login(key=os.getenv("WANDB_API_KEY"))

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
  "microsoft/deberta-v3-base",
  num_labels=1
)
model.to('cuda')

print("Loading dataset...")
dataset = load_from_disk('/workspace/data/reddit/submissions/RS_2023-01-dataset')

# dataset.rename_column('log_score', 'labels')

print(dataset)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base', use_fast=True)

args = TrainingArguments(
  evaluation_strategy = "steps",
  save_strategy="steps",
  eval_steps=500,
  learning_rate=1e-5,
  per_device_train_batch_size=16,
  per_device_eval_batch_size=32,
  num_train_epochs=1,
  report_to="wandb",
  weight_decay=1e-6,
  output_dir="/workspace/models/reddit/RS_2023-01",
  save_total_limit=4,
  # metric_for_best_model="rmse",
  group_by_length=True,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
  model=model,
  args=args,
  train_dataset=dataset["train"],
  eval_dataset=dataset["eval"],
  tokenizer=tokenizer,
  data_collator=data_collator,
)

wandb.init(project="reddit-log-score", job_type='train')

# Test on the eval dataset to start
print("Testing...")
trainer.evaluate()

print("Training...")
trainer.train()