#!/usr/bin/env python
# coding: utf-8

# In[2]:

# The relevance prediction is optional.
# It finetunes a BERT classifier to predict how likely a news article is related to the CP5 or not.
# The training is using a simple assumption: the news artilces with URL linked to at least one tweet (that tweet is labeled with some frmae) is relevant.
# This file only contains the inference step.

import pytorch_lightning as pl
import os
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import AlbertTokenizer, AlbertModel, AlbertConfig
from transformers import AdamW
from transformers import get_constant_schedule_with_warmup
from transformers import get_linear_schedule_with_warmup
import transformers
from tqdm import tqdm
import numpy as np
import argparse
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_logger
from sklearn.metrics import f1_score, precision_score, recall_score
from pytorch_lightning.metrics import F1, Precision, Recall

seed_everything(42)


def encoding_text(d):
    inputs = tokenizer.encode(d, return_tensors="pt", truncation=True, padding='max_length', max_length=256)
    return inputs.squeeze()


class BertClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.cls = transformers.BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased')
        self.softmax = torch.nn.Softmax(dim=-1)
        self.crossentropy = torch.nn.CrossEntropyLoss()
        self.f1 = F1()
        self.p = Precision()
        self.r = Recall()

    def forward(self, x):
        return self.softmax(self.cls(x).logits)

    def training_step(self, batch, batch_idx):
        x, label = batch
        pred = self.forward(x)
        loss = self.crossentropy(pred, label)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, label = batch
        logits = self.forward(x)
        return label, logits[:,1]
    
    def evaluate_outs(self, outs):
        labels = torch.cat([label for label, pred in outs])
        preds = torch.cat([pred for label, pred in outs])
        f1 = self.f1(preds, labels)
        p = self.p(preds, labels)
        r = self.r(preds, labels)
        return f1, p, r
    
    def validation_epoch_end(self, outs):
        f1, p, r = self.evaluate_outs(outs)
        print('f1', f1)
        print('p', p)
        print('r', r)
        self.log("val_f1", f1)
        self.log("val_p", p)
        self.log("val_r", r)
        return f1
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def test_epoch_end(self, outs):
        f1, p, r = self.evaluate_outs(outs)
        self.log("test_f1", f1)
        self.log("test_p", p)
        self.log("test_r", r)
        return f1

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
        return [optimizer], [scheduler]

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

import json
class binary_dataset(Dataset):
    def __init__(self, fn, maximum_training=-1):
        self.X = []
        self.Y = []
        with open(fn) as fin:
            for line in fin:
                js = json.loads(line)
                self.Y.append(js['label'])
                self.X.append(js['text'])
                if maximum_training != -1 and len(self.X) > maximum_training:
                    break
    def __getitem__(self, idx):
        return encoding_text(self.X[idx]), int(self.Y[idx])
    def __len__(self):
        return len(self.X)
best_threshold = 4.1355447e-06
# generate relevance score for all news
model = BertClassifier.load_from_checkpoint('./relevance_model_best_f1.ckpt')
class inference_dataset(Dataset):
    def __init__(self):
        self.X = []
        with open('../news_text_raw_append.json') as fin:
            for line in fin:
                i = json.loads(line)
                text = (i['title'] if i['title'] else ' ') + ' ' + (i['article'] if i['article'] else ' ')
                self.X.append(text)
    def __getitem__(self, idx):
        return encoding_text(self.X[idx])
    def __len__(self):
        return len(self.X)
inference_set = inference_dataset()
inference_loader = DataLoader(inference_set, batch_size=256, num_workers=128, shuffle=False, drop_last=False)
relevance_scores = []
relevance_predictions = []
with torch.no_grad():
    model.eval().to('cuda')
    for batch in tqdm(inference_loader):
        tmp = model(batch.to('cuda'))[:,1].detach().cpu().numpy()
        relevance_scores.extend(tmp)
        relevance_predictions.extend(tmp >= best_threshold)
all_data = []
with open('../news_text_raw_append.json') as fin:
    for line in fin:
        i = json.loads(line)
        all_data.append(i)
url2gt = dict()
with open('tweet_news_text_for_relevance.json') as fin:
    for line in fin:
        js = json.loads(line)
        url2gt[js['url']] = js['label']
with open('manual_news_text_for_relevance.json') as fin:
    for line in fin:
        js = json.loads(line)
        url2gt[js['url']] = js['label']
assert len(all_data) == len(relevance_scores)
with open('url2relevance_append.json', 'w') as fout:
    for js, score, pred in zip(all_data, relevance_scores, relevance_predictions):
        if js['url'] in url2gt:
            js['relevance_prediction'] = url2gt[js['url']]
            js['relevance_score'] = 1.0 if js['relevance_prediction'] else 0.0
        else:
            js['relevance_score'] = float(score)
            js['relevance_prediction'] = bool(pred)
        fout.write(json.dumps({'url': js['url'], 'relevance_score': js['relevance_score'], 'relevance_prediction': js['relevance_prediction']}) + '\n')
        
cnt = 0
for js in all_data:
    if js['relevance_prediction'] and js['lang'] == 'en':
        cnt += 1
print(cnt)    

