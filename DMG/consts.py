import utils
from typing import Tuple


class ClassifierConfig:
    def __init__(self, cls_name, path_url2frame2prob, hard_threshold, soft_threshold):
        self.name = cls_name
        self.hard_threshold = hard_threshold  # > hard threshold to assign infoids (hard label)
        self.soft_threshold = soft_threshold  # > soft threshold to be accumulated in the curve
        self.path_url2frame2prob = path_url2frame2prob
        self.url2frame2prob = utils.Json.load(path_url2frame2prob)


class Config:
    def __init__(self, name, path_raw_news_json, path_phrased_news_json):
        self.name = name
        self.path_raw_news_json = path_raw_news_json
        self.path_phrased_news_json = path_phrased_news_json
        self.frames: Tuple[str] = (
            'covid', 'covid/assistance', 'debt', 'environmentalism', 'infrastructure', 'mistreatment', 'prejudice', 'travel', 'un'
        )

        self.output_dir = utils.IO.mkdir(self.name)
        self.path_merged = self.output_dir / 'merged_hybrid.jsonl'

        self.classifier_configs = [
            ClassifierConfig(
                cls_name='retrieval',
                path_url2frame2prob='./data/retrieved_url2frame2prob_append.json',
                hard_threshold=0.1,
                soft_threshold=0.5
            ),
            ClassifierConfig(
                cls_name='Leidos',
                path_url2frame2prob='./data/frame/url2roberta_prob_append.json',
                hard_threshold=0.1,
                soft_threshold=0.5
            ),

            ClassifierConfig(
                cls_name='Leidos+retrieval_ft2',
                path_url2frame2prob='./data/ft_retrieval_roberta_epoch2_append.json',
                hard_threshold=0.1,
                soft_threshold=0.5
            ),
            ClassifierConfig(
                cls_name='Westclass_top1000',
                path_url2frame2prob='./data/ft_top1000_retrieval_westclass_append.json',
                hard_threshold=0.1,
                soft_threshold=0.2
            ),
            ClassifierConfig(
                cls_name='Westclass',
                path_url2frame2prob='./data/ft_retrieval_westclass_append.json',
                hard_threshold=0.1,
                soft_threshold=0.2
            ),

        ]
        self.cls2config = {conf.name: conf for conf in self.classifier_configs}


config = Config(
    name='v4_append',
    path_raw_news_json='../data/NewsArticles/cp6.ea.newsarticles.training.v1.json',
    path_phrased_news_json='../data/0830appended/ucphrase_eval_append.json',
)

'''

            ClassifierConfig(
                cls_name='hybrid3',
                path_url2frame2prob='../frame_classification/LeidosRoberta/frame/normed_url2roberta_prob_append.json',
                hard_threshold=0.1,
                soft_threshold=0.5
            ),
'''