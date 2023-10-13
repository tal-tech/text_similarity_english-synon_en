# -*- coding: utf-8 -*-

import sys
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from model.Tal_sim_eng import qiao_infer
from preprocess import sentence
from unique_token import Constants
import json

# load path
basePath = os.path.dirname(os.path.realpath(__file__))
workplace = basePath

# load deep learning model

# device is cpu if no gpu available
device = torch.device("cpu")
# load vocabulary(a binary file)
vocabulary = torch.load(os.path.join(workplace,"bin","word_vocab"))

# load model parameters
# which we can tune
n_block_1 = 2
#n_block_2 = 2
n_head = 4
dff = 1280
dropout = 1
n_epoch = 100
lr = 1
bs = 64
# no change
d_model = 300
model = qiao_infer(d_model=d_model, number_block_1=n_block_1, head_number=n_head, d_ff=dff, seq_len=30,vocab_size=len(vocabulary),drop_out=dropout)
model.load_state_dict(torch.load(os.path.join(workplace,"bin","model_epoch_63.pth"),map_location=device))
model.eval()

def convert_instance_to_idx_seq(tokenized_words, word2idx):
    ''' Mapping words to idx sequence. '''
    return [word2idx.get(w, Constants.UNK) for w in tokenized_words]


class simEngCheck():
    def __init__(self,deep_model=model,vocab=vocabulary,lemma=False, stem=False,lower=False):
        self.sentence_1 = sentence(lemma=lemma, stem=stem, lower=lower)
        self.sentence_2 = sentence(lemma=lemma, stem=stem, lower=lower)
        self.model = deep_model
        self.vocab = vocab

    def forward(self, sentence_1, sentence_2):
        tokenized_left, signal_left = self.sentence_1.tokenize_sentence(sentence_1)
        tokenized_right, signal_right = self.sentence_2.tokenize_sentence(sentence_2)
        numeric_left = torch.LongTensor(convert_instance_to_idx_seq(tokenized_left, self.vocab)).to(device)
        numeric_right = torch.LongTensor(convert_instance_to_idx_seq(tokenized_right, self.vocab)).to(device)
        numeric_left = numeric_left.view(1, -1)
        numeric_right = numeric_right.view(1, -1)
        pred_prob = self.model(numeric_left, numeric_right)
        one_item = {'text1': " ".join(tokenized_left) if signal_left==0 else sentence_1,
                    'text2': " ".join(tokenized_right) if signal_right==0 else sentence_2,
                    'similarity': pred_prob.item(), }

        return json.dumps(one_item, ensure_ascii=False)

