import utils
import torch
import datalib
import modellib
import argparse
import numpy as np
from tqdm import tqdm
from transformers.tokenization_xlm_roberta import XLMRobertaTokenizer
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler


def parse_args():
    parser = argparse.ArgumentParser(description='Apply SocialSim BERT Models')
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--model', type=str, default='../../../cea_frame_model/pytorch_model.bin')
    parser.add_argument('--dir_data', type=str, default='../retrieved_docs/')
    return parser.parse_args()


@torch.no_grad()
def predict(model: modellib.RobertaForMultiLabelSequenceClassification, corpus: datalib.Corpus, device, batch_size):
    model.eval()
    tokenizer = XLMRobertaTokenizer(vocab_file='../../../cea_frame_model/sentencepiece.bpe.model')
    print('load tokenizer OK')

    docs = list(corpus.url2doc.values())[:100]
    test_features = [doc.to_feature(tokenizer) for doc in docs]

    print("***** Running prediction *****")
    print("  Num examples = %d", len(test_features))
    all_input_ids = torch.tensor([f['input_ids'] for f in test_features], dtype=torch.long)
    all_input_masks = torch.tensor([f['input_masks'] for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f['segment_ids'] for f in test_features], dtype=torch.long)

    test_data = TensorDataset(all_input_ids, all_input_masks, all_segment_ids)

    # Run prediction for full data
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    all_logits = list()
    for step, batch in enumerate(tqdm(test_dataloader, desc="Prediction Iteration")):
        input_ids, input_mask, segment_ids = batch
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        logits = model(input_ids, segment_ids, input_mask)
        logits = logits.sigmoid().detach().cpu().numpy().tolist()
        all_logits += logits

    results = []
    assert len(all_logits) == len(docs)
    for doc, logits in zip(docs, all_logits):
        result = {
            'url': doc.url,
            'frames': list(doc.frames),
            'frame2prob': {datalib.Doc.id2frame[i]: logit for i, logit in enumerate(logits)}
        }
        results.append(result)
    return results


if __name__ == '__main__':
    CARGS = parse_args()

    ''' get corpus '''
    corpus = datalib.Corpus(CARGS.dir_data)
    print('get corpus ok')

    ''' get model '''
    device = torch.device(f'cuda:{CARGS.gpu}' if CARGS.gpu is not None else 'cpu')
    model_state_dict = torch.load(CARGS.model, map_location=torch.device('cpu'))
    model = modellib.RobertaForMultiLabelSequenceClassification.from_pretrained('xlm-roberta-large', num_labels=10, state_dict=model_state_dict).to(device)
    print('load model ok')

    results = predict(
        model=model,
        corpus=corpus,
        device=device,
        batch_size=512)
    utils.Json.dump(results, './tmp_results.json')