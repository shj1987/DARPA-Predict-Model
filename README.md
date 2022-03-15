# DARPA-Predict-Model
Prediction models for CP6

## Create the environment
```
conda create -n ssenv python=3.8
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Pipeline
The pipeline is created with CP6 data as the `DATA_ROOT`. The input to our pipeline includes the raw exogenous data and extracted social platform ground truth. We also expect to have the RoBERTa PyTorch frame classification model be saved in `DMG/cea_frame_model`. Then, the following could be run:
```
cd misc
sh timeseries.sh
sh corr.sh

cd ../DMG
sh classification_pipeline.sh
cp v1_append/* $WORK_DIR

cd ../misc
sh entropy.sh

cd ../LR_PLUS
sh run.sh

cd ../LR
sh run.sh

cd ../DT_Model
sh run.sh

cd ../misc
sh fill.sh
```

## New dataset
To adapt a new dataset, a new set of information IDs have to be defined. These files should be modified:
- Node file
- `DMG/WeSTClass/news_manual/classes.txt and keywords.txt (index does not matter, order should match dimensions defined in the classification model provided)
- `DMG/retrieval/index.py`: `frame2keywords` dictionary
- `DMG/roberta/frame/tmp/classes.txt`

## Dataset pre-checks
In order for most of the codes to run smoothly, we expect the following basic properties of the dataset:
- A dozen events for each frame (information ID).
- A dozen users for each frame.
- Not all events are of the same timestamp.

## Miscellaneous
We will perform user id substitution during the last filling step. If you need users in a frame remains in the set of training data users, you could put those frames in the preserve node list.

DMG's code will take massive amount of memory if there are many news articles.

There are always some special cases for a dataset, please contact us if there are exceptions arising from the codebase.
