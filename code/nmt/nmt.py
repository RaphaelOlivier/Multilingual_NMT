import sys
import time
import os

import numpy as np
import torch
from typing import List
from docopt import docopt

from utils import read_corpus
from vocab import Vocab, VocabEntry, MultipleVocab
from nmt.nmtmodel import NMTModel
import config
from nmt import routine
import paths
from nmt import pretrain


def train(helper=False):
    print_file = sys.stderr
    if config.printout:
        print_file = sys.stdout
    train_data_src = read_corpus(paths.train_source, source='src')
    train_data_tgt = read_corpus(paths.train_target, source='tgt')

    dev_data_src = read_corpus(paths.dev_source, source='src')
    dev_data_tgt = read_corpus(paths.dev_target, source='tgt')

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    train_batch_size = config.batch_size
    valid_niter = config.valid_niter
    log_every = config.log_every
    model_save_path = paths.model(helper=False)
    max_epoch = config.max_epoch

    if config.sanity:
        log_every = 1
        train_data = train_data[:150]
        dev_data = dev_data[:150]
        max_epoch = 2
    pretraining = config.pretraining
    if config.load:
        try:
            model = NMTModel.load(model_save_path)
            pretraining = False
        except:
            print("Impossible to load the model ; creating a new one.")
            model = NMTModel(helper=False)
    else:
        model = NMTModel()

    if config.cuda:
        model.to_gpu()
    else:
        print("No cuda support")

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cumulative_tgt_words = report_tgt_words = 0
    cumulative_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    lr = config.lr
    max_patience = config.patience
    max_num_trial = config.max_num_trial
    lr_decay = config.lr_decay

    if pretraining:
        #print("Pretraining the encoder")
        #pretrain.train_encoder(model, train_data, dev_data)
        print("Pretraining the decoder")
        pretrain.train_decoder(model, train_data, dev_data, model_save_path,
                               train_batch_size, valid_niter, log_every, config.max_epoch_pretraining, lr, max_patience, max_num_trial, lr_decay)

    model = routine.train_model(model, train_data, dev_data, model_save_path,
                                train_batch_size, valid_niter, log_every, max_epoch, lr, max_patience, max_num_trial, lr_decay)
    exit(0)


def decode(helper=False):
    """
    performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    """
    if config.test:
        data_src = read_corpus(paths.test_source, source='src')
        data_tgt = read_corpus(paths.test_target, source='tgt')
        data_tgt_path = paths.test_target
    else:
        data_src = read_corpus(paths.dev_source, source='src')
        data_tgt = read_corpus(paths.dev_target, source='tgt')
        data_tgt_path = paths.dev_target

    print(f"load model from {paths.model}", file=sys.stderr)
    model = NMTModel.load(paths.model(helper=helper))
    if config.cuda:
        model.to_gpu()
    model.eval()
    max_step = None
    if config.sanity:
        max_step = 3

    hypotheses = routine.beam_search(model, data_src, max_step, replace=config.replace)

    if config.target_in_decode:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        #bleu_score = routine.compute_corpus_level_bleu_score(data_tgt, top_hypotheses)
        #print(f'Corpus BLEU: {bleu_score}', file=sys.stderr)

    with open(paths.decode_output, 'w') as f:
        for src_sent, hyps in zip(data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp.value)
            f.write(hyp_sent + '\n')

    bleu_command = "perl scripts/multi-bleu.perl "+data_tgt_path+" < "+paths.decode_output
    os.system(bleu_command)