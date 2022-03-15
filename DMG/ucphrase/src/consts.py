import torch
import transformers

# huggingface LM
GPT_TOKEN = 'Ġ'
LM_NAME = 'roberta-base'
LM_NAME_SUFFIX = LM_NAME.split('/')[-1]

# html visualization
HTML_BP = '<span style=\"color: #ff0000\">'
HTML_EP = '</span>'

# settings
MAX_SENT_LEN = 64
MAX_WORD_GRAM = 5
MAX_SUBWORD_GRAM = 10
NEGATIVE_RATIO = 1

# multiprocessing
NUM_CORES = 16
torch.set_num_threads(NUM_CORES)

LM_TOKENIZER = transformers.RobertaTokenizerFast.from_pretrained(LM_NAME)


def roberta_tokens_to_str(tokens):
    return ''.join(tokens).replace(GPT_TOKEN, ' ').strip()
