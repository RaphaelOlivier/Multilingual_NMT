import sys
import time
import os

import numpy as np
import torch
from typing import List
from docopt import docopt

from utils import read_corpus, zip_data
from vocab import Vocab, VocabEntry, MultipleVocab
from char.charmodel import CharModel
import char.charconfig as config
from nmt import routine
import paths


def train():
    print_file = sys.stderr
    if config.printout:
        print_file = sys.stdout
    train_data_src = read_corpus(paths.train_source, source='src', char=True)
    train_data_tgt = read_corpus(paths.train_target, source='tgt', char=True)

    dev_data_src = read_corpus(paths.dev_source, source='src', char=True)
    dev_data_tgt = read_corpus(paths.dev_target, source='tgt', char=True)

    train_data_src_helper = read_corpus(paths.train_source_helper, source='src', char=True)
    train_data_tgt_helper = read_corpus(paths.train_target_helper, source='tgt', char=True)

    dev_data_src = read_corpus(paths.dev_source, source='src', char=True)
    dev_data_tgt = read_corpus(paths.dev_target, source='tgt', char=True)

    train_data = zip_data(train_data_src, train_data_tgt, "low",
                          train_data_src_helper, train_data_tgt_helper, "helper")
    dev_data = zip_data(dev_data_src, dev_data_tgt, "low")

    train_batch_size = config.batch_size
    valid_niter = config.valid_niter
    log_every = config.log_every
    model_save_path = paths.model(helper=False) + ".char"
    max_epoch = config.max_epoch

    if config.sanity:
        log_every = 1
        train_data = {train_data[key]: train_data[key][:150] for key in train_data.keys()}
        dev_data = {dev_data[key]: dev_data[key][:150] for key in dev_data.keys()}
        max_epoch = 2
    pretraining = config.pretraining
    pretraining_encoder = config.pretraining_encoder
    if config.load:
        try:
            model = CharModel.load(model_save_path)
            pretraining = False
            pretraining_encoder = False
        except:
            print("Impossible to load the model ; creating a new one.")
            model = CharModel()
    else:
        model = CharModel()

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

    if pretraining_encoder:
        #print("Pretraining the encoder")
        #pretrain.train_encoder(model, train_data, dev_data)
        print("Loading monolingual data")
        mono_data_src = read_corpus(paths.data_monolingual)
        mono_data_tgt = [[] for i in range(len(mono_data_src))]
        #train_helper_src = read_corpus(paths.train_source_helper)
        #train_helper_tgt = [[] for i in range(len(train_helper_src))]
        source_data = zip_data(mono_data_src, mono_data_tgt, "mono",
                               train_data_src, train_data_tgt, "low")
        print("Pretraining the encoder")
        routine.train_encoder(model, source_data, dev_data, model_save_path,
                              config.mono_batch_size, valid_niter, log_every, config.max_epoch_pretraining_encoder, lr, max_patience, max_num_trial, lr_decay)

    if pretraining:
        #print("Pretraining the encoder")
        #pretrain.train_encoder(model, train_data, dev_data)
        print("loading all target data")
        #target_data_tgt = []
        # for lg in config.all_languages:
        #    target_data_tgt = target_data_tgt + \
        #        read_corpus(paths.get_data_path(set="train", mode="tg", lg=lg))
        train_helper_tgt = read_corpus(paths.train_target_helper)
        train_helper_src = [[] for i in range(len(train_helper_tgt))]

        target_data = zip_data(train_helper_src, train_helper_tgt, "one")
        print("Pretraining the decoder")
        routine.train_decoder(model, target_data, dev_data, model_save_path,
                              train_batch_size, valid_niter, log_every, config.max_epoch_pretraining, lr, max_patience, max_num_trial, lr_decay)

    model = routine.train_model(model, train_data, dev_data, model_save_path,
                                train_batch_size, valid_niter, log_every, max_epoch, lr, max_patience, max_num_trial, lr_decay)
    model.to_cpu()
    exit(0)


def decode():
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

    print(f"load model from {paths.model(helper=False)}.shared", file=sys.stderr)
    model = CharModel.load(paths.model(helper=False) + ".shared")
    if config.cuda:
        model.to_gpu()
    model.eval()
    max_step = None
    if config.sanity:
        max_step = 3

    hypotheses = routine.beam_search(
        model, data_src, max_step=max_step, key="low", replace=config.replace)

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


if __name__ == '__main__':
    train()
