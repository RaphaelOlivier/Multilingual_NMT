import sys
import time
import os

import numpy as np
import torch
from typing import List
from docopt import docopt

from utils import read_corpus, write_sents
from vocab import Vocab, VocabEntry, MultipleVocab
from nmt import routine
import paths
from transfer.transfermodel import TransferModel
import transfer.transferconfig as config


def train(load_helper=False, train_helper=False, load=True):
    if load_helper:
        print("Loading model trained on helper language :", config.helper_language)
    else:
        print("Loading model trained on low-ressource language :", config.language)

    print_file = sys.stderr
    if config.printout:
        print_file = sys.stdout

    train_batch_size = config.batch_size
    valid_niter = config.valid_niter(train_helper)
    log_every = config.log_every(train_helper)

    max_epoch = config.max_epoch(train_helper)

    if config.sanity:
        log_every = 1
        max_epoch = 2
    pretraining = config.pretraining
    if load_helper:
        model_save_path = paths.model(helper=True) + ".transfer"
    else:
        model_save_path = paths.model(helper=False) + ".transfer"

    if load:
        try:
            model = TransferModel.load(model_save_path, helper=load_helper)
            pretraining = False
        except:
            print("Impossible to load the model ; creating a new one.")
            model = TransferModel()
    else:
        model = TransferModel()

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

    if train_helper:
        print("Training model on helper language :", config.helper_language)
        model_save_path = paths.model(helper=True) + ".transfer"
        if not load_helper:
            print("Transfering model from low-resource language to helper language")
            model.switch()
            # model.save(model_save_path)
    else:
        print("Training model on low-ressource language :", config.language)
        model_save_path = paths.model(helper=False) + ".transfer"
        pretraining = False
        if load_helper:
            print("Transfering model from helper language to low-resource language")
            model.switch()
            # model.save(model_save_path)

    if train_helper:
        train_data_src = read_corpus(paths.train_source_helper, source='src')
        train_data_tgt = read_corpus(paths.train_target_helper, source='tgt')

        dev_data_src = read_corpus(paths.dev_source_helper, source='src')
        dev_data_tgt = read_corpus(paths.dev_target_helper, source='tgt')
    else:
        train_data_src = read_corpus(paths.train_source, source='src')
        train_data_tgt = read_corpus(paths.train_target, source='tgt')

        dev_data_src = read_corpus(paths.dev_source, source='src')
        dev_data_tgt = read_corpus(paths.dev_target, source='tgt')

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    if config.sanity:
        train_data = train_data[:150]
        dev_data = dev_data[:150]

    if pretraining:
        #print("Pretraining the encoder")
        #pretrain.train_encoder(model, train_data, dev_data)
        print("Pretraining the decoder")
        routine.train_decoder(model, train_data, dev_data, model_save_path,
                              train_batch_size, valid_niter, log_every, config.max_epoch_pretraining, lr, max_patience, max_num_trial, lr_decay)

    model = routine.train_model(model, train_data, dev_data, model_save_path,
                                train_batch_size, valid_niter, log_every, max_epoch, lr, max_patience, max_num_trial, lr_decay)


def decode(decode_helper=False):
    """
    performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    """
    if decode_helper:
        print("Decoding helper language :", config.helper_language)

    else:
        print("Decoding low-resource language :", config.language)

    if config.test:
        if decode_helper:
            data_src = read_corpus(paths.test_source_helper, source='src')
            data_tgt = read_corpus(paths.test_target_helper, source='tgt')
            data_tgt_path = paths.test_target_helper
        else:
            data_src = read_corpus(paths.test_source, source='src')
            data_tgt = read_corpus(paths.test_target, source='tgt')
            data_tgt_path = paths.test_target
    else:
        if decode_helper:
            data_src = read_corpus(paths.dev_source_helper, source='src')
            data_tgt = read_corpus(paths.dev_target_helper, source='tgt')
            data_tgt_path = paths.dev_target_helper
        else:
            data_src = read_corpus(paths.dev_source, source='src')
            data_tgt = read_corpus(paths.dev_target, source='tgt')
            data_tgt_path = paths.dev_target

    print(f"load model from {paths.model(helper=decode_helper)}", file=sys.stderr)
    model_path = paths.model(helper=decode_helper) + ".transfer"
    model = TransferModel.load(model_path, helper=decode_helper)
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

    lines = []
    for src_sent, hyps in zip(data_src, hypotheses):
        top_hyp = hyps[0]
        lines.append(top_hyp.value)
    write_sents(lines, paths.decode_output)

    bleu_command = "perl scripts/multi-bleu.perl "+data_tgt_path+" < "+paths.decode_output
    os.system(bleu_command)
