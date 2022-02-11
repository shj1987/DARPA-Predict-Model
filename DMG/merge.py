import utils
import consts
from tqdm import tqdm
from typing import List


def merge(classifier_configs: List[consts.ClassifierConfig]):
    docs = utils.JsonLine.load(consts.config.path_phrased_news_json)
    for doc in tqdm(docs, ncols=100, desc='merging'):
        for result in classifier_configs:
            url = doc['url']
            cls_name = result.name
            doc[f'{cls_name}_infoids'] = []
            if url in result.url2frame2prob:
                frame2prob = result.url2frame2prob[url]
                if 'assistance' in frame2prob:
                    frame2prob['covid/assistance'] = frame2prob['assistance']
                    frame2prob.pop('assistance')
                doc[f'{cls_name}_prob'] = frame2prob
                for frame, prob in frame2prob.items():
                    if frame == 'date':
                        continue
                    if prob > result.soft_threshold:
                        doc[f'{cls_name}_infoids'].append(frame)
    return docs


if __name__ == '__main__':
    utils.JsonLine.dump(merge(consts.config.classifier_configs), consts.config.path_merged)