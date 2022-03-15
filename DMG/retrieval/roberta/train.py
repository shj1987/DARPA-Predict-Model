import ipdb
import torch
import utils
import datalib
import modellib
import argparse
from tqdm import tqdm
from torch.optim import Adam
from transformers.tokenization_xlm_roberta import XLMRobertaTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Apply SocialSim BERT Models')
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--articles', type=str, required=True)
    parser.add_argument('--dir_data', type=str, default=Path(__file__).parent.joinpath('..', 'retrieved_docs'))
    parser.add_argument('--model', type=str, default=Path(__file__).parent.joinpath('..', '..', 'cea_frame_model', 'pytorch_model.bin'))
    parser.add_argument('--tokenizer', type=str, default=Path(__file__).parent.joinpath('..', '..', 'cea_frame_model', 'sentencepiece.bpe.model'))
    return parser.parse_args()


class Trainer:
    def __init__(self):
        self.args = parse_args()
        self.device = utils.get_device(self.args.gpu)
        self.train_dataloader = self.get_train_dataloader()

        print('loading model', end='...')
        model_state_dict = torch.load(self.args.model, map_location=torch.device('cpu'))
        # model_state_dict = None
        self.model = modellib.RobertaForMultiLabelSequenceClassification.from_pretrained('xlm-roberta-large', num_labels=10, state_dict=model_state_dict).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=1e-5)
        print('OK!')

    def get_train_dataloader(self):
        print('loading tokenizer', end='...')
        tokenizer = XLMRobertaTokenizer(vocab_file=self.args.tokenizer)
        print('OK!')

        print('loading training data', end='...')
        corpus = datalib.Corpus(self.args.dir_data, self.args.articles)
        pos_train_docs = list(corpus.url2doc.values())
        neg_train_docs = corpus.sample_negative_docs()
        train_docs = pos_train_docs + neg_train_docs
        train_features = [doc.to_feature(tokenizer) for doc in train_docs]
        all_input_ids = torch.tensor([f['input_ids'] for f in train_features], dtype=torch.long)
        all_input_masks = torch.tensor([f['input_masks'] for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f['segment_ids'] for f in train_features], dtype=torch.long)
        all_multihot_labels = torch.tensor([f['multihot_labels'] for f in train_features], dtype=torch.float)
        train_data = TensorDataset(all_input_ids, all_input_masks, all_segment_ids, all_multihot_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.args.batch_size)
        print('OK!')

        return train_dataloader

    def train(self, num_epochs=5):
        for epoch in range(1, num_epochs + 1):
            utils.Log.info(f'Epoch [{epoch} / {num_epochs}]')

            # Train
            epoch_loss = 0
            self.model.train()
            for input_ids, input_mask, segment_ids, multihot_labels in tqdm(self.train_dataloader, desc='training'):
                self.model.zero_grad()
                input_ids = input_ids.to(self.device)
                input_mask = input_mask.to(self.device)
                segment_ids = segment_ids.to(self.device)
                multihot_labels = multihot_labels.to(self.device)
                # ipdb.set_trace()
                batch_loss = self.model(input_ids, segment_ids, input_mask, labels=multihot_labels)

                # print(batch_loss.item())
                epoch_loss += batch_loss.item()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            train_loss = epoch_loss / len(self.train_dataloader)
            utils.Log.info(f'Train loss: {train_loss}')
            torch.save(self.model.state_dict(), f'./retrieval/roberta/ft-roberta-epoch{epoch}.ckpt')


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()