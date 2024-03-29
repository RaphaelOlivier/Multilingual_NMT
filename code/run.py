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

import warnings
warnings.filterwarnings("ignore")

from docopt import docopt
import numpy as np
import torch
import config
from nmt import nmt
from shared import shared
from transfer import transfer
from transfer import transferconfig
from multi import multi
from vocab import Vocab, VocabEntry, MultipleVocab


def simple_script(args):
    if args['train']:
        nmt.train()
    elif args['decode']:
        nmt.decode()
    else:
        raise RuntimeError(f'invalid command')


def multi_script(args):
    if args['train']:
        multi.train()
    elif args['decode']:
        multi.decode()
    else:
        raise RuntimeError(f'invalid command')


def shared_script(args):
    if args['train']:
        shared.train()
    elif args['decode']:
        shared.decode()
    else:
        raise RuntimeError(f'invalid command')


def transfer_script(args):
    if args['train']:
        if transferconfig.load_helper_model and transferconfig.train_helper_model:
            transfer.train(load_helper=True, train_helper=True, load=config.load)
            transfer.train(load_helper=True, train_helper=False, load=True)
        elif (not transferconfig.load_helper_model) and (not transferconfig.train_helper_model):
            transfer.train(load_helper=False, train_helper=False)
        elif transferconfig.load_helper_model and (not transferconfig.train_helper_model):
            transfer.train(load_helper=True, train_helper=False)
        else:
            transfer.train(load_helper=False, train_helper=True)
    elif args['decode']:
        if transferconfig.decode_helper_model:
            transfer.decode(decode_helper=True)
        else:
            transfer.decode(decode_helper=False)

    else:
        raise RuntimeError(f'invalid command')


def main():
    args = docopt(__doc__)

    # seed the random number generator (RNG), you may
    # also want to seed the RNG of tensorflow, pytorch, dynet, etc.
    seed = int(config.seed)
    np.random.seed(seed * 13 // 7)
    torch.manual_seed(seed * 13 // 7)
    if config.mode == "transfer":
        print("Transfer mode")
        transfer_script(args)
    elif config.mode == "multi":
        print("Multitask mode")
        multi_script(args)
    elif config.mode == "shared":
        print("Shared encoder mode")
        shared_script(args)
    else:
        print("Normal mode")
        simple_script(args)


if __name__ == '__main__':
    main()
