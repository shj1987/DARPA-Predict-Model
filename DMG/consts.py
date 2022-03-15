import utils
from typing import Tuple
from IPython import embed

class ClassifierConfig:
    def __init__(self, cls_name, path_url2frame2prob, hard_threshold, soft_threshold):
        self.name = cls_name
        self.hard_threshold = hard_threshold  # > hard threshold to assign infoids (hard label)
        self.soft_threshold = soft_threshold  # > soft threshold to be accumulated in the curve
        self.path_url2frame2prob = path_url2frame2prob
        self.url2frame2prob = {}
        for line in utils.JsonLine.load(path_url2frame2prob):
            self.url2frame2prob[line['url']] = line['prob']
        
        #embed()


class Config:
    def __init__(self, name, path_raw_news_json, path_phrased_news_json):
        self.name = name
        self.path_raw_news_json = path_raw_news_json
        self.path_phrased_news_json = path_phrased_news_json
        with open('WeSTClass/news_manual/classes.txt') as IN:
            classes = []
            for line in IN:
                classes.append(line.strip().split(':')[1])
        self.frames: Tuple[str] = classes

        self.output_dir = utils.IO.mkdir(self.name)
        self.path_merged = self.output_dir / 'merged_hybrid.jsonl'

        self.classifier_configs = [
            ClassifierConfig(
                cls_name='Leidos',
                path_url2frame2prob='./roberta/frame/prob_append_eval4_cp6.ea.newsarticles.twitter.2021-01-11_2021-01-31.json',
                hard_threshold=0.0,
                soft_threshold=0.0
            ),
            # ClassifierConfig(
            #     cls_name='Leidos+retrieval_ft2',
            #     path_url2frame2prob='./roberta/frame/prob_append_ft2_eval4_cp6.ea.newsarticles.twitter.2021-01-11_2021-01-31.json',
            #     hard_threshold=0.0,
            #     soft_threshold=0.0
            # ),
            ClassifierConfig(
                cls_name='Westclass',
                path_url2frame2prob='./data/ft_retrieval_westclass_append.json',
                hard_threshold=0.0,
                soft_threshold=0.0
            ),

        ]
        self.cls2config = {conf.name: conf for conf in self.classifier_configs}


