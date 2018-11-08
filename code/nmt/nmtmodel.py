
import pickle
import time

import numpy as np
from typing import List

import torch
import torch.nn as nn

from utils import Hypothesis, batch_iter
from nmt.encoder import Encoder
from nmt.decoder import Decoder
import config
import paths

from allennlp.modules.elmo import batch_to_ids


class NMTModel:

    def __init__(self, helper=False):
        self.helper = helper
        self.vocab = pickle.load(open(paths.vocab, 'rb'))
        self.encoder = Encoder(len(self.vocab.src(self.helper)))
        self.decoder = Decoder(len(self.vocab.tgt(self.helper)))
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.params = list(self.encoder.parameters())+list(self.decoder.parameters())
        self.optimizer = torch.optim.Adam(
            self.params, lr=config.lr, weight_decay=config.weight_decay)
        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.decoder_optimizer = torch.optim.Adam(
            self.decoder.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.gpu = False
        self.initialize()

        # initialize neural network layers...
    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def to_gpu(self):
        if not self.gpu:
            self.gpu = True
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()

    def to_cpu(self):
        if self.gpu:
            self.gpu = False
            self.encoder = self.encoder.cpu()
            self.decoder = self.decoder.cpu()

    def train(self):
        self.encoder = self.encoder.train()
        self.decoder = self.decoder.train()

    def eval(self):
        self.encoder = self.encoder.eval()
        self.decoder = self.decoder.eval()

    def __call__(self, src_sents: List[List[str]], tgt_sents: List[List[str]], update_params=True):
        """
        take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences.

        Args:
            src_sents: list of source sentence tokens
            tgt_sents: list of target sentence tokens, wrapped by `<s>` and `</s>`

        Returns:
            scores: a variable/tensor of shape (batch_size, ) representing the
                log-likelihood of generating the gold-standard target sentence for
                each example in the input batch
        """

        src_encodings, decoder_init_state = self.encode(src_sents)
        loss = self.decode(src_encodings, decoder_init_state, tgt_sents)
        """
        loss = torch.zeros(1)
        if self.gpu:
            loss = loss.cuda()
        for i in range(len(targets)):
            loss += self.criterion(preds[i][:-1], targets[i][1:])
        """
        if update_params:
            self.step(loss)
        if self.gpu:
            loss = loss.cpu()
        return loss.detach().numpy()

    def encode(self, src_sents: List[List[str]]):
        """
        Use a GRU/LSTM to encode source sentences into hidden states

        Args:
            src_sents: list of source sentence tokens

        Returns:
            src_encodings: hidden states of tokens in source sentences, this could be a variable
                with shape (batch_size, source_sentence_length, encoding_dim), or in orther formats
            decoder_init_state: decoder GRU/LSTM's initial state, computed from source encodings
        """

        tensor_sents = batch_to_ids(src_sents)
        # np_sents = [np.array([self.vocab.src(self.helper)[word] for word in sent])
        #             for sent in src_sents]
        # tensor_sents = [torch.LongTensor(s) for s in np_sents]
        if self.gpu:
            # tensor_sents = [x.cuda() for x in tensor_sents]
            tensor_sents = tensor_sents.cuda()
        src_encodings, decoder_init_state = self.encoder(tensor_sents)

        return src_encodings, decoder_init_state

    def decode(self, src_encodings, decoder_init_state, tgt_sents: List[List[str]]):
        """
        Given source encodings, compute the log-likelihood of predicting the gold-standard target
        sentence tokens

        Args:
            src_encodings: hidden states of tokens in source sentences
            decoder_init_state: decoder GRU/LSTM's initial state
            tgt_sents: list of gold-standard target sentences, wrapped by `<s>` and `</s>`

        Returns:
            scores: could be a variable of shape (batch_size, ) representing the
                log-likelihood of generating the gold-standard target sentence for
                each example in the input batch
        """
        np_sents = [np.array([self.vocab.tgt(self.helper)[word] for word in sent])
                    for sent in tgt_sents]
        tensor_sents = [torch.LongTensor(s) for s in np_sents]
        if self.gpu:
            tensor_sents = [x.cuda() for x in tensor_sents]
        loss = self.decoder(src_encodings, decoder_init_state, tensor_sents)

        return loss

    def decode_to_loss(self, tgt_sents, update_params=True):
        np_sents = [np.array([self.vocab.tgt(self.helper)[word] for word in sent])
                    for sent in tgt_sents]
        tensor_sents = [torch.LongTensor(s) for s in np_sents]
        if self.gpu:
            tensor_sents = [x.cuda() for x in tensor_sents]
        loss = self.decoder(None, None, tensor_sents, attend=False)
        """
        loss = torch.zeros(1)
        if self.gpu:
            loss = loss.cuda()
        for i in range(len(tensor_sents)):
            loss += self.criterion(preds[i][:-1], tensor_sents[i][1:])
        """
        if update_params:
            loss.backward()
            self.decoder_optimizer.step()
            self.decoder_optimizer.zero_grad()
        return loss.cpu().detach().numpy()

    def beam_search(self, src_sent: List[str], max_step=None, replace=False) -> List[Hypothesis]:
        """
        Given a single source sentence, perform beam search

        Args:
            src_sent: a single tokenized source sentence
            max_decoding_time_step: maximum number of time steps to unroll the decoding RNN

        Returns:
            hypotheses: a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """

        tensor_sent = batch_to_ids([src_sent])
        if self.gpu:
            tensor_sent = tensor_sent.cuda()
        src_encoding, decoder_init_state = self.encoder.encode_one_sent(tensor_sent)
        if config.greedy_search:
            tgt_tensor, score = self.decoder.greedy_search(
                src_encoding, decoder_init_state, max_step, replace=replace)
            tgt_np = tgt_tensor.cpu().detach().numpy()
            tgt_sent = []
            for i in tgt_np[1:-1]:
                if i >= 0:
                    tgt_sent.append(self.vocab.tgt(self.helper).id2word[i])
                else:
                    tgt_sent.append(src_sent[-i])
            hypotheses = [Hypothesis(tgt_sent, score)]

        else:
            l = self.decoder.beam_search(
                src_encoding, decoder_init_state, max_step, replace=replace)
            hypotheses = []
            for tgt_tensor, score in l:
                tgt_np = tgt_tensor.cpu().detach().numpy()
                tgt_sent = []
                for i in tgt_np[1: -1]:
                    if i > 0:
                        tgt_sent.append(self.vocab.tgt(self.helper).id2word[i])
                    else:
                        tgt_sent.append(src_sent[-i])
                hypotheses.append(Hypothesis(tgt_sent, score))
        return hypotheses

    def evaluate_ppl(self, dev_data, batch_size: int=32, encoder_only=False, decoder_only=False):
        """
        Evaluate perplexity on dev sentences

        Args:
            dev_data: a list of dev sentences
            batch_size: batch size

        Returns:
            ppl: the perplexity on dev sentences
        """

        cum_loss = 0.
        cum_tgt_words = 0.

        # you may want to wrap the following code using a context manager provided
        # by the NN library to signal the backend to not to keep gradient information
        # e.g., `torch.no_grad()`

        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            if encoder_only:
                loss = self.encode_to_loss(src_sents, update_params=False)
            elif decoder_only:
                loss = self.decode_to_loss(tgt_sents, update_params=False)
            else:
                loss = self(src_sents, tgt_sents, update_params=False)
            cum_loss += loss
            tgt_word_num_to_predict = sum(len(s[1:])
                                          for s in tgt_sents)  # omitting the leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

        return ppl

    def step(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    @staticmethod
    def load(model_path: str):
        """
        Load a pre-trained model

        Returns:
            model: the loaded model
        """
        enc_path = model_path+".enc.pt"
        dec_path = model_path+".dec.pt"
        model = NMTModel()
        model.encoder.load_state_dict(torch.load(enc_path))
        model.decoder.load_state_dict(torch.load(dec_path))

        return model

    def initialize(self):

        for param in self.encoder.parameters():
            torch.nn.init.uniform_(param, -0.1, 0.1)
        for param in self.decoder.parameters():
            torch.nn.init.uniform_(param, -0.1, 0.1)

    def load_params(self, model_path):
        if model_path is None:
            model_path = paths.model(helper=self.helper)
        enc_path = model_path + ".enc.pt"
        dec_path = model_path + ".dec.pt"
        opt_path = model_path + ".opt.pt"
        self.encoder.load_state_dict(torch.load(enc_path))
        self.decoder.load_state_dict(torch.load(dec_path))
        self.optimizer.load_state_dict(torch.load(opt_path))

    def save(self, path: str):
        """
        Save current model to file
        """
        enc_path = path+".enc.pt"
        dec_path = path+".dec.pt"
        opt_path = path+".opt.pt"
        torch.save(self.encoder.state_dict(), enc_path)
        torch.save(self.decoder.state_dict(), dec_path)
        torch.save(self.optimizer.state_dict(), opt_path)
