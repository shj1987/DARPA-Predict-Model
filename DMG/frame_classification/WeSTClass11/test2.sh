export CUDA_VISIBLE_DEVICES=1

#  --trained_weights /home/dmg/CP4/news_match/WeSTClass/results/news/cnn/phase3/final.h5

/home/qiuwenda/anaconda3/envs/west/bin/python main.py --dataset news_manual_full --sup_source keywords --model cnn --with_evaluation False
