import utils
import consts
import random
from tqdm import tqdm
from pathlib import Path
from data_lib.preprocess import Preprocessor


class BaseAnnotator:
    def __init__(self, preprocessor: Preprocessor, use_cache=True):
        self.use_cache = use_cache
        self.preprocessor = preprocessor
        self.dir_output = self.preprocessor.dir_preprocess / f'annotate.{self.__class__.__name__}'
        self.dir_output.mkdir(exist_ok=True)
        self.path_tokenized_corpus = self.preprocessor.path_tokenized_corpus
        self.path_tokenized_id_corpus = self.preprocessor.path_tokenized_id_corpus
        self.path_marked_corpus = self.dir_output / f'{self.path_tokenized_corpus.name}'

    @staticmethod
    def _par_sample_train_data(marked_doc):
        sents = marked_doc['sents']
        for sent in sents:
            phrases = sent['phrases']
            assert phrases
            positive_spans = [tuple(phrase[0]) for phrase in phrases]
            num_positive = len(positive_spans)
            # sample negatives
            word_idxs = sent['widxs']
            all_spans = utils.get_possible_spans(word_idxs, len(sent['ids']), consts.MAX_WORD_GRAM, consts.MAX_SUBWORD_GRAM)
            possible_negative_spans = set(all_spans) - set(positive_spans)
            num_negative = min(len(possible_negative_spans), int(num_positive * consts.NEGATIVE_RATIO))
            sampled_negative_spans = random.sample(possible_negative_spans, k=num_negative)
            sent['pos_spans'] = positive_spans
            sent['neg_spans'] = sampled_negative_spans
            sent.pop('phrases')
        return marked_doc

    def sample_train_data(self):
        assert utils.IO.is_valid_file(self.path_marked_corpus)

        path_output = self.dir_output / f'sampled.neg{consts.NEGATIVE_RATIO}.{self.path_marked_corpus.name}'
        if self.use_cache and utils.IO.is_valid_file(path_output):
            print(f'[SampleTrain] Use cache: {path_output}')
            return path_output

        marked_docs = utils.JsonLine.load(self.path_marked_corpus)
        sampled_docs = [BaseAnnotator._par_sample_train_data(d) for d in tqdm(marked_docs, ncols=100, desc='[Sample Train]')]
        utils.JsonLine.dump(sampled_docs, path_output)
        return path_output

    def _mark_corpus(self):
        raise NotImplementedError

    def mark_corpus(self):
        if self.use_cache and utils.IO.is_valid_file(self.path_marked_corpus):
            print(f'[Annotate] Use cache: {self.path_marked_corpus}')
            return
        marked_corpus = self._mark_corpus()
        # Remove empty sents and docs
        for raw_id_doc in marked_corpus:
            for sent in raw_id_doc['sents']:
                sent['phrases'] = [p for p in sent['phrases'] if p[0][1] - p[0][0] + 1 <= consts.MAX_SUBWORD_GRAM]
            raw_id_doc['sents'] = [s for s in raw_id_doc['sents'] if s['phrases']]
        marked_corpus = [d for d in marked_corpus if d['sents']]
        utils.JsonLine.dump(marked_corpus, self.path_marked_corpus)
