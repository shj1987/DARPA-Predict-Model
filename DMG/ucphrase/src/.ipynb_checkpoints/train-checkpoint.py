import utils
import consts
import data_lib
import model_lib
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--path_train", type=str, default='../data/devdata/devdata.train.jsonl')
    parser.add_argument("--dir_output", type=str, default='../train_output/devdata')
    parser.add_argument("--lm_layers", type=int, default=3, help='number of Transformer layers used')
    args = parser.parse_args()
    return args


class Controller:

    def __init__(self):
        self.args = parse_args()
        self.device = utils.get_device(self.args.gpu)
        self.rootdir = utils.IO.mkdir(self.args.dir_output)

        self.train_preprocessor = data_lib.Preprocessor(
            path_corpus=self.args.path_train,
            num_cores=consts.NUM_CORES,
            use_cache=True)

        self.train_annotator = data_lib.CoreAnnotator(
            use_cache=True,
            preprocessor=self.train_preprocessor)
        
        model_dir = self.rootdir / 'model'
        model = model_lib.AttmapModel(
            device=self.device,
            model_dir=model_dir,
            max_num_subwords=consts.MAX_SUBWORD_GRAM,
            num_BERT_layers=self.args.lm_layers)
        self.trainer = model_lib.AttmapTrainer(model=model)

    def train(self, max_epochs=20, least_epochs=10):
        self.train_preprocessor.tokenize_corpus()
        self.train_annotator.mark_corpus()
        path_sampled_train_data = self.train_annotator.sample_train_data()
        self.trainer.train(path_sampled_train_data=path_sampled_train_data, max_epochs=max_epochs, least_epochs=least_epochs)


if __name__ == '__main__':
    controller = Controller()
    controller.train()
