import os
import sys
import random
import logging
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from pathlib import Path
from tqdm import tqdm
from transformers.tokenization_xlm_roberta import XLMRobertaTokenizer
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

import pickle
import ipdb
import torch
import datalib
import modellib


parser = argparse.ArgumentParser(description='Apply SocialSim BERT Models')
parser.add_argument('--gpu', type=int, default=None, help='which gpu to use')
parser.add_argument('--model', type=str, default='./cea_frame_model/pytorch_model.bin', help='Path to model .bin file')
parser.add_argument('--tokenizer', type=str, default='./cea_frame_model/sentencepiece.bpe.model', help='Path to model .bin file')
parser.add_argument('--type', type=str, choices=['frame', 'stance'], default='frame', help='Either the string "frame" for frame or "stance" for stance')
parser.add_argument('--input', type=str, default='./data/eval3_cp6.ea.newsarticles.youtube.2020-12-21_2021-01-10.csv', help='Input csv file with "comment" and "id" columns')
parser.add_argument('--dir', type=str, default='./roberta/frame', help='Directory containing /tmp/classes.txt for either stance or frame')
parser.add_argument('--output', type=str, default='./roberta/frame/ft_retrieval_roberta.csv', help='Output csv file path')

CARGS = parser.parse_args()


@torch.no_grad()
def predict(model, args, dir_data, device, test_filename='test.csv'):
    model.eval()
    predict_processor = datalib.MultiLabelTextProcessor(dir_data)
    test_examples = predict_processor.get_test_examples(dir_data, test_filename, size=-1)
    print('load test examples ok')

    label_list = datalib.MultiLabelTextProcessor(dir_data / 'tmp').get_labels()
    tokenizer = XLMRobertaTokenizer(vocab_file=CARGS.tokenizer)
    print('load tokenizer ok')

    input_data = [{'id': input_example.guid, 'comment_text': input_example.text_a} for input_example in test_examples]
    test_features = datalib.convert_examples_to_features(
        test_examples, label_list, args['max_seq_length'], tokenizer)
#     with open('./tmp_cached_test_features.pk', 'wb') as wf:
#         pickle.dump(test_features, wf)
#     with open('./tmp_cached_test_features.pk', 'rb') as rf:
#         test_features = pickle.load(rf)

    print("***** Running prediction *****")
    print("  Num examples = %d", len(test_examples))
    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)

    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    
    # Run prediction for full data
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args['eval_batch_size'])

    all_logits = None
    
    nb_eval_steps, nb_eval_examples = 0, 0

    for step, batch in enumerate(tqdm(test_dataloader, desc="Prediction Iteration")):
        input_ids, input_mask, segment_ids = batch
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)
            logits = logits.sigmoid()

        if all_logits is None:
            all_logits = logits.detach().cpu().numpy()
        else:
            all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)
            
        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    return pd.merge(pd.DataFrame(input_data), pd.DataFrame(all_logits, columns=label_list), left_index=True,
                    right_index=True)


if __name__ == '__main__':
    ''' prepare folders'''
    dir_data = Path(CARGS.dir)
    dir_data_tmp = dir_data / 'tmp'
    dir_data_cls = dir_data_tmp / 'class'
    dir_data_cls.mkdir(exist_ok=True, parents=True)

    ''' prepare args '''
    args = {
        "train_size": 4133,
        "val_size": 459,
        "full_data_dir": dir_data,
        "data_dir": dir_data_tmp,
        "task_name": "bert_model",
        "no_cuda": False,
        'bert_model': 'xlm-roberta-large',
        # 'bert_model': 'bert-base-multilingual-cased',
        "output_dir": dir_data_cls / 'output',
        "max_seq_length": 100,
        "do_train": False,
        "do_eval": True,
        "do_lower_case": False,
        "train_batch_size": 32,
        # "eval_batch_size": 32,
        "eval_batch_size": 256,
        "learning_rate": 3e-5,
        "num_train_epochs": 50.0,
        "warmup_proportion": 0.1,
        "local_rank": -1,
        "seed": 42,
        "gradient_accumulation_steps": 1,
        "optimize_on_cpu": False,
        "fp16": False,
        "loss_scale": 128,
        'output_hidden_states': False
    }

#     test_filename = 'cur_batch.csv'
    test_filename = CARGS.input.split('/')[-1].replace('.csv', '_batch.csv')

    ''' prepare data '''
    temp = pd.read_csv(CARGS.input)
    temp.to_csv(dir_data / test_filename, index=False)
    print('load data ok')

    ''' get model '''
    device = torch.device(f'cuda:{CARGS.gpu}' if CARGS.gpu is not None else 'cpu')
    model_state_dict = torch.load(CARGS.model, map_location=torch.device('cpu'))
    model = modellib.RobertaForMultiLabelSequenceClassification.from_pretrained(args['bert_model'], num_labels=10, state_dict=model_state_dict).to(device)
    print('load model ok')

    ''' predict '''
    result = predict(model, args, dir_data, device, test_filename=test_filename)
    result.to_csv(CARGS.output, index=False)