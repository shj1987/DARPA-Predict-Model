Dependencies:
	See requirements.txt

Main Script:
	classification_pipeline.sh

ucphrase/
    code
        ucphrase/src/
    data
        ucphrase/train_output/kpTimes/model/
        ucphrase/stopwords.txt
    Output
        $UCPHRASE_OUTPUT_PATH in classification_pipeline.sh

WeSTclass/
    data
        news_manual/classes.txt (frame names)
        news_manual/keywords.txt (keywords for each frame)
    Output
        (default) data/ft_retrieval_westclass_append.json


roberta/
    code
        roberta/*.py
    data
        roberta/frame/tmp/classes.txt  (frame names)
        cea_frame_model/
    Output
        $path_output_append in classification_roberta.sh (pre-trained roberta) and classification_retrieval.sh (fine-tuned roberta)

retrieval
    code
        retrieval/*.py
        retrieval/roberta/*.py
        customize frame names and corresponding keywords for retrieval:
        retrieval/consts.py (note: the keys should be aligned with roberta/frame/tmp/classes.txt)
    Output files
        ./retrieval/retrieved_docs_eval3-youtube/
