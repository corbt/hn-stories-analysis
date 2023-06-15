from dotenv import load_dotenv

import random
import os
import torch
import torch.nn as nn
from transformers import (
    AutoModel, AutoConfig, 
    AutoTokenizer, logging,
    AdamW, get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
    Trainer, TrainingArguments
)
# from datasets import Dataset, DatasetDict
from transformers.modeling_outputs import SequenceClassifierOutput
import pandas as pd
import numpy as np
import wandb

load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))

logging.set_verbosity_error()
logging.set_verbosity_warning()

CONFIG = {
    "model_name": "microsoft/deberta-v3-base",# "distilbert-base-uncased",
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    "dropout": random.uniform(0.01, 0.60),
    "max_length": 512,
    "train_batch_size": 8,
    "valid_batch_size": 16,
    "epochs": 10,
    "folds" : 3,
    "max_grad_norm": 1000,
    "weight_decay": 1e-6,
    "learning_rate": 1e-5,
     "loss_type": "rmse",
    "n_accumulate" : 1,
    "label_cols" : ['upvote_ratio', 'log_score'],   
}

print("Loading dataset...")
train = pd.read_feather('/workspace/data/reddit/submissions/RS_2023-01-train.arrow')
test = pd.read_feather('/workspace/data/reddit/submissions/RS_2023-01-test.arrow')

class CustomIterator(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, labels=CONFIG['label_cols'], is_train=True):
        self.df = df
        self.tokenizer = tokenizer
        self.max_seq_length = CONFIG["max_length"]# tokenizer.model_max_length
        self.labels = labels
        self.is_train = is_train
        
    def __getitem__(self,idx):
        tokens = self.tokenizer(
                    self.df.loc[idx, 'formatted_text'],#.to_list(),
                    add_special_tokens=True,
                    padding='max_length',
                    max_length=self.max_seq_length,
                    truncation=True,
                    return_tensors='pt',
                    return_attention_mask=True
                )     
        res = {
            'input_ids': tokens['input_ids'].to(CONFIG.get('device')).squeeze(),
            'attention_mask': tokens['attention_mask'].to(CONFIG.get('device')).squeeze()
        }
        
        if self.is_train:
            res["labels"] = torch.tensor(
                self.df.loc[idx, self.labels].to_list(), 
            ).to(CONFIG.get('device')) 
            
        return res
    
    def __len__(self):
        return len(self.df)
      
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()  
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

class FeedBackModel(nn.Module):
    def __init__(self, model_name):
        super(FeedBackModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.hidden_dropout_prob = 0
        self.config.attention_probs_dropout_prob = 0
        self.model = AutoModel.from_pretrained(model_name, config=self.config)
        self.drop = nn.Dropout(p=0.2)
        self.pooler = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, len(CONFIG['label_cols']))
        
    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids,
                         attention_mask=attention_mask, 
                         output_hidden_states=False)
        out = self.pooler(out.last_hidden_state, attention_mask)
        out = self.drop(out)
        outputs = self.fc(out)
        return SequenceClassifierOutput(logits=outputs)

class RMSELoss(nn.Module):
    """
    Code taken from Y Nakama's notebook (https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-train)
    """
    def __init__(self, reduction='mean', eps=1e-9):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.reduction = reduction
        self.eps = eps

    def forward(self, predictions, targets):
        loss = torch.sqrt(self.mse(predictions, targets) + self.eps)
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
        loss_func = RMSELoss(reduction='mean')
        loss = loss_func(outputs.logits.float(), inputs['labels'].float())
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    colwise_rmse = np.sqrt(np.mean((labels - predictions) ** 2, axis=0))
    res = {
        f"{analytic.upper()}_RMSE" : colwise_rmse[i]
        for i, analytic in enumerate(CONFIG["label_cols"])
    }
    res["MCRMSE"] = np.mean(colwise_rmse)
    return res

# set seed to produce similar folds
SEED = 1318

train = train.reset_index(drop=True)

training_args = TrainingArguments(
        output_dir="outputs/",
        evaluation_strategy="steps",
        save_steps=500,
        per_device_train_batch_size=CONFIG['train_batch_size'],
        per_device_eval_batch_size=CONFIG['valid_batch_size'],
        num_train_epochs=CONFIG['epochs'],
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        gradient_accumulation_steps=CONFIG['n_accumulate'],
        seed=SEED,
        group_by_length=True,
        max_grad_norm=CONFIG['max_grad_norm'],
        metric_for_best_model='eval_MCRMSE',
        load_best_model_at_end=True,
        greater_is_better=False,
        save_strategy="steps",
        save_total_limit=3,
        report_to="wandb",
        label_names=["labels"]
    )

# Data Collator for Dynamic Padding

tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'], use_fast=True)
collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

wandb.init(project="reddit-deberta-v3", 
                  config=CONFIG,
                  job_type='train',
                  group="reddit-BASELINE-MODEL",
                  tags=[CONFIG['model_name'], CONFIG['loss_type'], "10-epochs"],
                  name='run-1',
                  anonymous='must')
# create iterators
train_dataset = CustomIterator(train, tokenizer)
valid_dataset = CustomIterator(test.sample(1000), tokenizer)
# init model
model = FeedBackModel(CONFIG['model_name'])
model.to(CONFIG['device'])

# SET THE OPITMIZER AND THE SCHEDULER
# no decay for bias and normalization layers
param_optimizer = list(model.named_parameters())
no_decay = ["bias", "LayerNorm.weight"]
optimizer_parameters = [
    {
        "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        "weight_decay": CONFIG['weight_decay'],
    },
    {
        "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer = AdamW(optimizer_parameters, lr=CONFIG['learning_rate'])
num_training_steps = (len(train_dataset) * CONFIG['epochs']) // (CONFIG['train_batch_size'] * CONFIG['n_accumulate'])
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0.1*num_training_steps,
    num_training_steps=num_training_steps
)
# CREATE THE TRAINER
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=collate_fn,
    optimizers=(optimizer, scheduler),
    compute_metrics=compute_metrics
)

print("Training model...")
trainer.train()