import sys
import time
import os

import numpy as np
import torch
from typing import List
from docopt import docopt

from utils import read_corpus, zip_data, write_sents
from vocab import Vocab, VocabEntry, MultipleVocab
from nmt import routine
import paths
from multi.multiwaymodel import MultiWayModel
import multi.multiconfig as config


def train():
    print_file = sys.stderr
    if config.printout:
        print_file = sys.stdout
    train_data_src_low = read_corpus(paths.train_source, source='src')
    train_data_tgt_low = read_corpus(paths.train_target, source='tgt')

    dev_data_src_low = read_corpus(paths.dev_source, source='src')
    dev_data_tgt_low = read_corpus(paths.dev_target, source='tgt')

    train_data_src_helper = read_corpus(paths.train_source_helper, source='src')
    train_data_tgt_helper = read_corpus(paths.train_target_helper, source='tgt')

    dev_data_src_helper = read_corpus(paths.dev_source_helper, source='src')
    dev_data_tgt_helper = read_corpus(paths.dev_target_helper, source='tgt')

    train_data = zip_data(train_data_src_low, train_data_tgt_low, "low",
                          train_data_src_helper, train_data_tgt_helper, "helper")

    train_data_low = zip_data(train_data_src_low, train_data_tgt_low, "low")
    train_data_helper = zip_data(train_data_src_helper, train_data_tgt_helper, "helper")

    dev_data_low = zip_data(dev_data_src_low, dev_data_tgt_low, "low")
    dev_data_helper = zip_data(dev_data_src_helper, dev_data_tgt_helper, "helper")

    train_batch_size = config.batch_size
    valid_niter = config.valid_niter
    log_every = config.log_every
    model_save_path = paths.model(helper=False) + ".multi"
    max_epoch = config.max_epoch
    sampling = config.sampling

    if config.sanity:
        log_every = 1
        valid_niter = 5
        train_data = dict([(k, v[:150]) for (k, v) in train_data.items()])
        dev_data_low = dict([(k, v[:150]) for (k, v) in dev_data_low.items()])
        dev_data_helper = dict([(k, v[:150]) for (k, v) in dev_data_helper.items()])
        train_data_low = dict([(k, v[:150]) for (k, v) in train_data_low.items()])
        train_data_helper = dict([(k, v[:150]) for (k, v) in train_data_helper.items()])
        max_epoch = 2
    pretraining_decoder = config.pretraining_decoder
    pretraining_encoders = config.pretraining_encoders
    if config.load:
        #model = MultiWayModel.load(model_save_path)

        try:
            model = MultiWayModel.load(model_save_path)
            pretraining_decoder = False
            pretraining_encoders = False
        except:
            print("Impossible to load the model ; creating a new one.")
            model = MultiWayModel()
    else:
        model = MultiWayModel()

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

    if pretraining_decoder:
        # print("Pretraining the encoder")
        # pretrain.train_encoder(model, train_data, dev_data)
        print("Pretraining the decoder")
        model.activate_discriminator = False
        routine.train_decoder(model, train_data_helper, dev_data_helper, model_save_path,
                              train_batch_size, valid_niter, log_every, config.max_epoch_pretraining_decoder, lr, max_patience, max_num_trial, lr_decay)
        routine.train_decoder(model, train_data_low, dev_data_low, model_save_path,
                              train_batch_size, valid_niter, log_every, config.max_epoch_pretraining_decoder, lr, max_patience, max_num_trial, lr_decay)

    if pretraining_encoders:
        # print("Pretraining the encoder")
        # pretrain.train_encoder(model, train_data, dev_data)
        model.activate_discriminator = False
        print("Pretraining the helper encoder")
        routine.train_model(model, train_data_helper, dev_data_helper, model_save_path,
                            train_batch_size, valid_niter, log_every, config.max_epoch_pretraining_helper, lr, max_patience, max_num_trial, lr_decay)
        print("Pretraining the low-resource encoder")
        routine.train_model(model, train_data_low, dev_data_low, model_save_path,
                            train_batch_size, valid_niter, log_every, config.max_epoch_pretraining_low, lr, max_patience, max_num_trial, lr_decay)

    print("Multitask training")
    model.activate_discriminator = True
    model = routine.train_model(model, train_data, dev_data_low, model_save_path,
                                train_batch_size, valid_niter, log_every, max_epoch, lr, max_patience, max_num_trial, lr_decay, sampling_multi=sampling)
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

    print(f"load model from {paths.model(helper=False)}.multi", file=sys.stderr)
    model = MultiWayModel.load(paths.model(helper=False)+".multi")
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
        # bleu_score = routine.compute_corpus_level_bleu_score(data_tgt, top_hypotheses)
        # print(f'Corpus BLEU: {bleu_score}', file=sys.stderr)

    lines = []
    for src_sent, hyps in zip(data_src, hypotheses):
        top_hyp = hyps[0]
        lines.append(top_hyp.value)
    write_sents(lines, paths.decode_output)

    bleu_command = "perl scripts/multi-bleu.perl "+data_tgt_path+" < "+paths.decode_output
    os.system(bleu_command)
