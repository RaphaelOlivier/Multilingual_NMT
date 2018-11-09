#!/usr/bin/env python
"""
Generate the vocabulary file for neural network training
A vocabulary file is a mapping of tokens to their indices

Usage:
    python vocab.py

Options specified in config.py, paths in paths.py
"""

from typing import List
from collections import Counter
from itertools import chain
from docopt import docopt
import pickle

from utils import read_corpus, input_transpose

import config
import paths


class VocabEntry(object):
    def __init__(self):
        self.word2id = dict()
        self.unk_id = 3
        self.word2id['<pad>'] = 0
        self.word2id['<s>'] = 1
        self.word2id['</s>'] = 2
        self.word2id['<unk>'] = 3

        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        return self.id2word[wid]

    def add(self, word):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def words2indices(self, sents):
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=2, mono_corpus=None):
        vocab_entry = VocabEntry()

        word_freq = Counter(chain(*corpus))

        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        mono_words = []
        print(
            f'number of word types in aligned data: {len(word_freq)}, number of word types w/ frequency >= {freq_cutoff}: {len(valid_words)}')
        if mono_corpus is not None:
            mono_freq = Counter(chain(*mono_corpus))
            mono_words = [w for w, v in mono_freq.items(
            ) if v >= freq_cutoff]
            print(
                f'number of word types in unsupervised data: {len(mono_freq)}, number of word types w/ frequency >= {freq_cutoff}: {len(mono_words)}')

        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
        top_k_words = top_k_words + \
            sorted(mono_words, key=lambda w: mono_freq[w], reverse=True)[:size]

        for word in top_k_words:
            vocab_entry.add(word)
            if len(vocab_entry) > size:
                break

        return vocab_entry


class Vocab(object):
    def __init__(self, src, tgt, mono, vocab_size, freq_cutoff):
        #assert len(src_sents) == len(tgt_sents)

        print('initialize source vocabulary ..')
        self.src = VocabEntry.from_corpus(src, vocab_size, freq_cutoff, mono_corpus=mono)

        if isinstance(tgt, list):
            print('initialize target vocabulary ..')
            self.tgt = VocabEntry.from_corpus(tgt, vocab_size, freq_cutoff)
        else:
            self.tgt = tgt

    def __repr__(self):
        return 'Vocab(source %d words, target %d words)' % (len(self.src), len(self.tgt))


class MultipleVocab(object):
    def __init__(self, vocabs):
        self.vocabs = vocabs

    def src(self, helper=False):
        if not helper:
            return self.vocabs[config.language].src
        else:
            return self.vocabs[config.helper_language].src

    def tgt(self, helper=False):
        if not helper:
            return self.vocabs[config.language].tgt
        else:
            return self.vocabs[config.helper_language].tgt


if __name__ == '__main__':
    all_corpuses = {}
    for lg in ["az", "be", "gl", "pt", "tr", "ru"]:
        all_corpuses[lg] = []
        sc = paths.get_data_path("train", "sc", lg=lg)
        tg = paths.get_data_path("train", "tg", lg=lg)
        print('read in source sentences: %s' % sc)
        print('read in target sentences: %s' % tg)

        src_sents = read_corpus(sc, source='src', char=True)
        tgt_sents = read_corpus(tg, source='tgt', char=True)
        mono_sents = None
        if config.vocab_mono and lg in ["az", "be", "gl"]:
            print('read in monolingual sentences')
            mono = paths.get_mono_path(lg)
            mono_sents = read_corpus(mono, source='src')

        all_corpuses[lg].append(src_sents)
        all_corpuses[lg].append(tgt_sents)
        all_corpuses[lg].append(mono_sents)

    if config.merge_target_vocab:
        all_tgt_sents = []
        for lg in ["az", "be", "gl", "pt", "tr", "ru"]:
            all_tgt_sents = all_tgt_sents + all_corpuses[lg][1]
        print('initialize target vocabulary ..')
        vocab_tgt = VocabEntry.from_corpus(all_tgt_sents, config.vocab_size, config.freq_cutoff)
    all_vocabs = {}

    for lg in ["az", "be", "gl", "pt", "tr", "ru"]:
        if config.merge_target_vocab:
            vocab = Vocab(all_corpuses[lg][0], vocab_tgt, all_corpuses[lg][2],
                          config.vocab_size, config.freq_cutoff)
        else:
            vocab = Vocab(all_corpuses[lg][0], all_corpuses[lg][1], all_corpuses[lg][2],
                          config.vocab_size, config.freq_cutoff)
        print('generated vocabulary, language %s,  source %d words, target %d words' %
              (lg, len(vocab.src), len(vocab.tgt)))
        all_vocabs[lg] = vocab

    if config.mode == "shared" or "char":
        for lg1, lg2 in [("az", "tr"), ("gl", "pt"), ("be", "ru")]:
            for word_helper in all_vocabs[lg2].src.word2id.keys():
                all_vocabs[lg1].src.add(word_helper)
            for word_helper in all_vocabs[lg2].tgt.word2id.keys():
                all_vocabs[lg1].tgt.add(word_helper)
            print("For shared mode, added helper words to lr vocab, new sizes",
                  len(all_vocabs[lg1].src), len(all_vocabs[lg1].tgt))

    main_vocab = MultipleVocab(all_vocabs)

    pickle.dump(main_vocab, open(paths.vocab, 'wb'))
    print('vocabulary saved to %s' % paths.vocab)
