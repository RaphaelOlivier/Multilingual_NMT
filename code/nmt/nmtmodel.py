
import pickle
import time

import numpy as np
from typing import List

import torch
import torch.nn as nn

from utils import Hypothesis, batch_iter, load_partial_state_dict
from nmt.encoder import Encoder
from nmt.decoder import Decoder
import config
import paths


class NMTModel:

    def __init__(self, helper=False, add_tokens_src=0):
        self.helper = helper
        self.vocab = pickle.load(open(paths.vocab, 'rb'))
        stproj = None if (config.hidden_size_encoder * (2 if config.bidirectional_encoder else 1)
                          == config.hidden_size_decoder) else config.hidden_size_decoder
        print(len(self.vocab.src(self.helper)))
        self.encoder = Encoder(len(self.vocab.src(self.helper))+add_tokens_src, config.embed_size, config.hidden_size_encoder, num_layers=config.num_layers_encoder,
                               bidirectional=config.bidirectional_encoder, dropout=config.dropout_layers, context_projection=None, state_projection=stproj)
        self.decoder = Decoder(len(self.vocab.tgt(self.helper)),
                               context_projection=config.hidden_size_decoder)
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.params = list(self.encoder.parameters())+list(self.decoder.parameters())
        self.optimizer = torch.optim.Adam(
            self.params, lr=config.lr, weight_decay=config.weight_decay)
        self.decoder_optimizer = torch.optim.Adam(
            self.decoder.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.gpu = False
        self.initialize()
        print(len(self.vocab.src(False)))
        # initialize neural network layers...

    def update_lr(self, lr, encoder=False, decoder=False, **kwargs):
        if encoder:
            opt = self.encoder_optimizer
        elif decoder:
            opt = self.decoder_optimizer
        else:
            opt = self.optimizer
        for param_group in opt.param_groups:
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

    def __call__(self, src_sents: List[List[str]], tgt_sents: List[List[str]], update_params=True, **kwargs):

        src_encodings, decoder_init_state = self.encode(src_sents)
        loss = self.decode(src_encodings, decoder_init_state, tgt_sents)

        if update_params:
            self.step(loss)
        if self.gpu:
            loss = loss.cpu()
        return loss.detach().numpy()

    def encode(self, src_sents: List[List[str]], **kwargs):

        np_sents = [np.array([self.vocab.src(self.helper)[word] for word in sent])
                    for sent in src_sents]
        # if config.flip_source:
        #    np_sents = [np.flip(a, axis=0).copy() for a in np_sents]

        tensor_sents = [torch.LongTensor(s) for s in np_sents]
        if self.gpu:
            tensor_sents = [x.cuda() for x in tensor_sents]
        src_encodings, decoder_init_state = self.encoder(tensor_sents)

        return src_encodings, decoder_init_state

    def decode(self, src_encodings, decoder_init_state, tgt_sents: List[List[str]]):

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

        if update_params:
            loss.backward()
            self.decoder_optimizer.step()
            self.decoder_optimizer.zero_grad()
        return loss.cpu().detach().numpy()

    def encode_to_loss(self, src_sents, update_params=True, **kwargs):
        # for s in src_sents:
        #    print(s)
        np_sents = [np.array([self.vocab.src(self.helper)[word] for word in sent])
                    for sent in src_sents]
        tensor_sents = [torch.LongTensor(s) for s in np_sents]
        if self.gpu:
            tensor_sents = [x.cuda() for x in tensor_sents]
        loss = self.encoder(tensor_sents, to_loss=True)

        if update_params:
            loss.backward()
            self.encoder_optimizer.step()
            self.encoder_optimizer.zero_grad()
        return loss.cpu().detach().numpy()

    def beam_search(self, src_sent: List[str], max_step=None, replace=False, **kwargs) -> List[Hypothesis]:

        np_sent = np.array([self.vocab.src(self.helper)[word] for word in src_sent])
        # if config.flip_source:
        #    np_sent = np.flip(np_sent, axis=0).copy()
        tensor_sent = torch.LongTensor(np_sent)
        if self.gpu:
            tensor_sent = tensor_sent.cuda()
        src_encoding, decoder_init_state = self.encoder.encode_one_sent(tensor_sent)
        #print(src_sent, np_sent, src_encoding.size())
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

    def evaluate_ppl(self, dev_data, batch_size: int=32, encoder_only=False, decoder_only=False, **kwargs):

        cum_loss = 0.
        cum_tgt_words = 0.

        # you may want to wrap the following code using a context manager provided
        # by the NN library to signal the backend to not to keep gradient information
        # e.g., `torch.no_grad()`
        if encoder_only:
            cum_src_words = 0.
            for src_sents, tgt_sents, key in batch_iter(dev_data, batch_size):
                loss = self.encode_to_loss(src_sents, update_params=False)

                src_word_num_to_predict = sum(len(s[1:])
                                              for s in src_sents)  # omitting the leading `<s>`
                cum_src_words += src_word_num_to_predict
                cum_loss += loss
            ppl = np.exp(cum_loss / cum_src_words)

            return ppl

        for src_sents, tgt_sents, key in batch_iter(dev_data, batch_size):

            if decoder_only:
                loss = self.decode_to_loss(tgt_sents, update_params=False)
            else:
                loss = self(src_sents, tgt_sents, key=key, update_params=False)
            cum_loss += loss
            tgt_word_num_to_predict = sum(len(s[1:])
                                          for s in tgt_sents)  # omitting the leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

        return ppl

    def step(self, loss, **kwargs):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    @staticmethod
    def load(model_path: str):

        enc_path = model_path+".enc.pt"
        dec_path = model_path+".dec.pt"
        model = NMTModel()
        print("Loading encoder")
        load_partial_state_dict(model.encoder, torch.load(enc_path))
        print("Loading decoder")
        load_partial_state_dict(model.decoder, torch.load(dec_path))

        return model

    def initialize(self):

        for name, param in self.encoder.named_parameters():
            torch.nn.init.uniform_(param, -0.1, 0.1)
        for name, param  in self.decoder.named_parameters():
            torch.nn.init.uniform_(param, -0.1, 0.1)

    def initialize_enc_embeddings(self, encoder_embeddings, freeze=True):

        self.encoder.lookup.weight.data.copy_(torch.from_numpy(encoder_embeddings))
        if freeze:
            self.encoder.lookup.weight.requires_grad = False

    def initialize_dec_embeddings(self, decoder_embeddings, freeze=True):

        self.decoder.lookup.weight.data.copy_(torch.from_numpy(decoder_embeddings))
        if freeze:
            self.decoder.lookup.weight.requires_grad = False

    def load_params(self, model_path):
        enc_path = model_path + ".enc.pt"
        dec_path = model_path + ".dec.pt"
        opt_path = model_path + ".opt.pt"
        self.encoder.load_state_dict(torch.load(enc_path))
        self.decoder.load_state_dict(torch.load(dec_path))
        self.optimizer.load_state_dict(torch.load(opt_path))

    def save(self, path: str):

        enc_path = path+".enc.pt"
        dec_path = path+".dec.pt"
        opt_path = path+".opt.pt"
        torch.save(self.encoder.state_dict(), enc_path)
        torch.save(self.decoder.state_dict(), dec_path)
        torch.save(self.optimizer.state_dict(), opt_path)
