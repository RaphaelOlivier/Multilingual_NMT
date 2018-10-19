# coding=utf-8

"""
A very basic implementation of neural machine translation

Usage:
    nmt.py train [options]
    nmt.py decode [options]
    nmt.py decode [options]

Options:
    -h --help                               show this screen.
"""

import math
import pickle
import sys
import time
import os

import numpy as np
import torch
from typing import List, Tuple, Dict, Set, Union
from docopt import docopt
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

from utils import read_corpus, batch_iter, Hypothesis
from vocab import Vocab, VocabEntry, MultipleVocab
from nmtmodel import NMTModel
import config
import paths
import pretrain


def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """
    Given decoding results and reference sentences, compute corpus-level BLEU score

    Args:
        references: a list of gold-standard reference target sentences
        hypotheses: a list of hypotheses, one for each reference

    Returns:
        bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]

    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])

    return bleu_score


def train():
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
    model_save_path = paths.model
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
            model = NMTModel()
    else:
        model = NMTModel()

    if config.cuda:
        model.to_gpu()
    else:
        print("No cuda support")

    if pretraining:
        #print("Pretraining the encoder")
        #pretrain.train_encoder(model, train_data, dev_data)
        print("Pretraining the decoder")
        pretrain.train_decoder(model, train_data, dev_data)

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cumulative_tgt_words = report_tgt_words = 0
    cumulative_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')
    lr = config.lr
    while True:
        epoch += 1
        model.train()
        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1
            batch_size = len(src_sents)

            # (batch_size)

            loss = model(src_sents, tgt_sents)

            report_loss += loss
            cum_loss += loss

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cumulative_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cumulative_examples += batch_size

            if train_iter % log_every == 0:
                log = 'epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f '\
                    'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                       report_loss / report_examples,
                                                                                       math.exp(
                                                                                           report_loss / report_tgt_words),
                                                                                       cumulative_examples,
                                                                                       report_tgt_words /
                                                                                       (time.time(
                                                                                       ) - train_time),
                                                                                       time.time() - begin_time)
                print(log, file=print_file)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # the following code performs validation on dev set, and controls the learning schedule
            # if the dev score is better than the last check point, then the current model is saved.
            # otherwise, we allow for that performance degeneration for up to `--patience` times;
            # if the dev score does not increase after `--patience` iterations, we reload the previously
            # saved best model (and the state of the optimizer), halve the learning rate and continue
            # training. This repeats for up to `--max-num-trial` times.
            if train_iter % valid_niter == 0:
                model.eval()
                log = 'epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                             cum_loss / cumulative_examples,
                                                                                             np.exp(
                                                                                                 cum_loss / cumulative_tgt_words),
                                                                                             cumulative_examples)
                print(log, file=print_file)
                cum_loss = cumulative_examples = cumulative_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=print_file)

                # compute dev. ppl and bleu
                # dev batch size can be a bit larger
                dev_ppl = model.evaluate_ppl(dev_data, batch_size=128)
                valid_metric = -dev_ppl

                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=print_file)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' %
                          model_save_path, file=print_file)
                    model.save(model_save_path)

                    # You may also save the optimizer's state
                elif patience < config.patience:
                    patience += 1
                    print('hit patience %d' % patience, file=print_file)

                    if patience == config.patience:
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=print_file)
                        if num_trial == config.max_num_trial:
                            print('early stop!', file=print_file)
                            exit(0)

                        # decay learning rate, and restore from previously best checkpoint
                        lr = lr * config.lr_decay

                        print('load previously best model and decay learning rate to %f' %
                              lr, file=print_file)
                        model.load_params()
                        # load model
                        print('restore parameters of the optimizers', file=print_file)

                        model.update_lr(lr)
                        # You may also need to load the state of the optimizer saved before

                        # reset patience
                        patience = 0
                model.train()

        if epoch == max_epoch:
            print('reached maximum number of epochs!', file=print_file)
            model.to_cpu()
            model.eval()
            model.save(model_save_path)
            exit(0)


def beam_search(model: NMTModel, test_data_src: List[List[str]], max_step=None, replace=False) -> List[List[Hypothesis]]:
    # was_training = model.training

    hypotheses = []
    for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
        example_hyps = model.beam_search(
            src_sent, max_step, replace=replace)

        hypotheses.append(example_hyps)

    return hypotheses


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

    print(f"load model from {paths.model}", file=sys.stderr)
    model = NMTModel.load(paths.model)
    if config.cuda:
        model.to_gpu()
    model.eval()
    max_step = None
    if config.sanity:
        max_step = 3

    hypotheses = beam_search(model, data_src, max_step, replace=config.replace)

    if config.target_in_decode:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        #bleu_score = compute_corpus_level_bleu_score(data_tgt, top_hypotheses)
        #print(f'Corpus BLEU: {bleu_score}', file=sys.stderr)

    with open(paths.decode_output, 'w') as f:
        for src_sent, hyps in zip(data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp.value)
            f.write(hyp_sent + '\n')

    bleu_command = "perl scripts/multi-bleu.perl "+data_tgt_path+" < "+paths.decode_output
    os.system(bleu_command)


def main():
    args = docopt(__doc__)

    # seed the random number generator (RNG), you may
    # also want to seed the RNG of tensorflow, pytorch, dynet, etc.
    seed = int(config.seed)
    np.random.seed(seed * 13 // 7)
    torch.manual_seed(seed * 13 // 7)

    if args['train']:
        train()
    elif args['decode']:
        decode()
    else:
        raise RuntimeError(f'invalid mode')


if __name__ == '__main__':
    main()
