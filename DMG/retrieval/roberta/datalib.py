import utils
import random
import transformers
from pathlib import Path


class Doc:
    max_seqlen = 100

    frames = [
        'covid',
        'assistance',
        'debt',
        'environmentalism',
        'infrastructure',
        'mistreatment',
        'prejudice',
        'out-of-topic',
        'travel',
        'un'
    ]
    negative_frame = 'out-of-topic'
    frame2id = {frame: i for i, frame in enumerate(frames)}
    id2frame = {i: frame for i, frame in enumerate(frames)}

    def __init__(self, doc_dict):
        self.frames = set()
        self.url = doc_dict['url']
        self.text = doc_dict['title'] + ' . ' + doc_dict['article']

    def to_feature(self, tokenizer: transformers.PreTrainedTokenizerFast):
        tokens = tokenizer.tokenize(self.text)
        tokens = tokens[:self.max_seqlen - 2]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]

        padding = [0] * (self.max_seqlen - len(input_ids))

        segment_ids = [0] * len(input_ids)
        input_masks = [1] * len(input_ids)
        input_ids += padding
        segment_ids += padding
        input_masks += padding
        assert len(input_ids) == len(segment_ids) == len(input_masks) == self.max_seqlen

        label_ids = [Doc.frame2id[frame] for frame in self.frames]
        multihot_labels = [0] * len(Doc.frame2id)
        for i in label_ids:
            multihot_labels[i] = 1
        return dict(input_ids=input_ids, segment_ids=segment_ids, input_masks=input_masks, multihot_labels=multihot_labels)


class Corpus:
    def __init__(self, dir_data, topk=3000):
        self.url2doc = dict()
        self.all_positive_urls = set()
        self.dir_data = Path(dir_data)

        for frame in Doc.frames:
            if frame != Doc.negative_frame:
                filepath = self.dir_data / (f'docs_{frame}.json' if frame != 'assistance' else 'docs_covid-assistance.json')
                doc_dicts = utils.Json.load(filepath)
                for doc_dict in doc_dicts[:topk]:
                    url = doc_dict['url']
                    if url not in self.url2doc:
                        self.url2doc[url] = Doc(doc_dict)
                    self.url2doc[url].frames.add(frame)
                for doc_dict in doc_dicts:
                    self.all_positive_urls.add(doc_dict['url'])

        self.all_doc_dicts = utils.JsonLine.load('./data/dmg_prob_append.json')
        self.negative_docs = []

    def sample_negative_docs(self):
        sampled_doc_dicts = random.sample(self.all_doc_dicts, k=len(self.url2doc))
        negative_doc_dicts = []
        for doc_dict in sampled_doc_dicts:
            if doc_dict['url'] in self.all_positive_urls:
                continue
            negative_doc = Doc(doc_dict)
            negative_doc.frames.add(Doc.negative_frame)
            negative_doc_dicts.append(negative_doc)
        return negative_doc_dicts