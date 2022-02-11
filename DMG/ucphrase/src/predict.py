import utils
import consts
import data_lib
import model_lib
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default='./ucphrase/train_output/kpTimes/model/epoch-14.ckpt')
    parser.add_argument("--threshold", type=float, default=0.85)
    # parser.add_argument("--path_corpus", type=str, default='../../21.socialsim/cp6/news_text_en_cleaned.json')
    # parser.add_argument("--dir_output", type=str, default='../decoded/cp6/news_text_en_cleaned_kptimes')
    parser.add_argument("--path_corpus", type=str, default='/shared/data2/qiz3/socialsim/data/0830appended/ucphrase_input_cleaned_eval3_cp6.ea.newsarticles.twitter.2020-12-21_2021-01-10.json')
    parser.add_argument("--dir_output", type=str, default='./ucphrase/decoded/cp6/cleaned_eval3_cp6.ea.newsarticles.twitter.2020-12-21_2021-01-10')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    device = utils.get_device(args.gpu)
    
    test_preprocessor = data_lib.Preprocessor(path_corpus=args.path_corpus, use_cache=True)
    test_preprocessor.tokenize_corpus()

    print('loading model...', end='')
    model = model_lib.AttmapModel.load_ckpt(device, args.ckpt).eval()
    print('OK!')
 
    dir_output = utils.IO.mkdir(args.dir_output)
    path_predicted_docs = model.predict(
        path_tokenized_id_corpus=test_preprocessor.path_tokenized_id_corpus, 
        dir_output=dir_output,
        device=device,
        batch_size=4096,
        use_cache=True)
    dir_decoded = utils.IO.mkdir(dir_output / 'decoded')
    path_decoded_doc2sents = model_lib.BaseModel.decode(
        path_predicted_docs=path_predicted_docs,
        output_dir=dir_decoded,
        threshold=args.threshold,
        use_cache=True,
        use_tqdm=True
    )
    print(f'Decoding finished: {path_decoded_doc2sents}')
