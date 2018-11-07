
import pickle
import time

import numpy as np
from typing import List

import torch
import torch.nn as nn

from utils import Hypothesis, batch_iter, load_partial_state_dict
from nmt.encoder import Encoder
from nmt.decoder import Decoder
import multi.multiconfig as config
import paths
from nmt.nmtmodel import NMTModel
import torch.nn as nn
from multi.discriminator import SequenceDiscriminator


class MultiWayModel(NMTModel):

    def __init__(self, *args, **kwargs):
        self.vocab = pickle.load(open(paths.vocab, 'rb'))
        self.keys = ["low", "helper"]
        self.inv_keys = {"low": config.smooth_disc, "helper": 1.-config.smooth_disc}
        self.encoder = self.create_two_encoders()
        self.vocab_src = self.create_two_vocabs()
        self.vocab_tgt = self.vocab.tgt(False)
        self.decoder = Decoder(len(self.vocab_tgt),
                               context_projection=config.context_size)
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.params = sum([list(self.encoder[k].parameters())
                           for k in self.encoder.keys()], [])+list(self.decoder.parameters())
        self.optimizer = torch.optim.Adam(
            self.params, lr=config.lr, weight_decay=config.weight_decay)
        self.decoder_optimizer = torch.optim.Adam(
            self.decoder.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        self.use_discriminator = config.discriminator
        if self.use_discriminator:
            self.discriminator = SequenceDiscriminator(
                sizes=[config.context_size, config.context_size, config.context_size, 1])
            self.discrim_optimizer = torch.optim.Adam(
                self.discriminator.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.gpu = False
        self.activate_discriminator = False
        self.train_discrim = 0
        self.train_not_discrim = 0
        self.cum_loss_mt = 0
        self.cum_loss_disc = 0
        self.initialize()

    def create_two_encoders(self):
        encoder_low = Encoder(len(self.vocab.src(helper=False)), config.embed_size_low, config.hidden_size_encoder_low, num_layers=config.num_layers_encoder_low,
                              bidirectional=config.bidirectional_encoder, dropout=config.dropout_layers, context_projection=config.context_size, state_projection=config.hidden_size_decoder)
        encoder_helper = Encoder(len(self.vocab.src(helper=True)), config.embed_size_helper, config.hidden_size_encoder_helper, num_layers=config.num_layers_encoder_helper,
                                 bidirectional=config.bidirectional_encoder, dropout=config.dropout_layers, context_projection=config.context_size, state_projection=config.hidden_size_decoder)
        return {"low": encoder_low, "helper": encoder_helper}

    def create_two_vocabs(self):
        return {"low": self.vocab.src(helper=False), "helper": self.vocab.src(helper=True)}

    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        if self.use_discriminator:
            for param_group in self.discrim_optimizer.param_groups:
                param_group['lr'] = lr

    def to_gpu(self):
        if not self.gpu:
            self.gpu = True
            for k in self.keys:
                self.encoder[k] = self.encoder[k].cuda()
            self.decoder = self.decoder.cuda()
            if self.use_discriminator:
                self.discriminator = self.discriminator.cuda()

    def to_cpu(self):
        if self.gpu:
            self.gpu = False
            for k in self.keys:
                self.encoder[k] = self.encoder[k].cpu()
            self.decoder = self.decoder.cpu()
            if self.use_discriminator:
                self.discriminator = self.discriminator.cpu()

    def train(self):
        for k in self.keys:
            self.encoder[k] = self.encoder[k].train()
        self.decoder = self.decoder.train()
        if self.use_discriminator:
            self.discriminator = self.discriminator.train()

    def eval(self):
        for k in self.keys:
            self.encoder[k] = self.encoder[k].eval()
        self.decoder = self.decoder.eval()
        if self.use_discriminator:
            self.discriminator = self.discriminator.eval()

    def __call__(self, src_sents, tgt_sents, key, update_params=True):
        src_encodings, decoder_init_state = self.encode(src_sents, key)
        bs = len(src_sents)
        # discriminator labels

        loss = self.decode(src_encodings, decoder_init_state, tgt_sents)
        adv_loss_disc = None
        adv_loss_mt = None
        no_disc_loss = loss
        if self.use_discriminator and self.activate_discriminator:
            key_labels = [[self.inv_keys[key] for w in s] for s in src_sents]
            key_labels = [torch.tensor(s) for s in key_labels]
            if self.gpu:
                key_labels = [x.cuda() for x in key_labels]
            adv_loss_mt, adv_loss_disc = self.discriminator(src_encodings, key_labels)
            loss = loss + config.adversarial_loss_coeff * adv_loss_mt
            if not update_params:
                self.cum_loss_disc += adv_loss_disc.detach().cpu().numpy()
                self.cum_loss_mt += adv_loss_mt.detach().cpu().numpy()

        if update_params:
            self.step(loss, loss_disc=adv_loss_disc, key=key)

        if self.gpu:
            no_disc_loss = no_disc_loss.cpu()
        return no_disc_loss.detach().numpy()

    def encode(self, src_sents: List[List[str]], key):
        np_sents = [np.array([self.vocab_src[key][word] for word in sent])
                    for sent in src_sents]
        # if config.flip_source:
        #    np_sents = [np.flip(a, axis=0).copy() for a in np_sents]

        tensor_sents = [torch.LongTensor(s) for s in np_sents]
        if self.gpu:
            tensor_sents = [x.cuda() for x in tensor_sents]
        src_encodings, decoder_init_state = self.encoder[key](tensor_sents)

        return src_encodings, decoder_init_state

    def decode(self, src_encodings, decoder_init_state, tgt_sents: List[List[str]]):

        np_sents = [np.array([self.vocab_tgt[word] for word in sent])
                    for sent in tgt_sents]
        tensor_sents = [torch.LongTensor(s) for s in np_sents]
        if self.gpu:
            tensor_sents = [x.cuda() for x in tensor_sents]
        loss = self.decoder(src_encodings, decoder_init_state, tensor_sents)

        return loss

    def decode_to_loss(self, tgt_sents, update_params=True):
        np_sents = [np.array([self.vocab_tgt[word] for word in sent])
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

    def beam_search(self, src_sent, key, max_step=None, replace=False) -> List[Hypothesis]:
        src_sent, key
        np_sent = np.array([self.vocab_src[key][word] for word in src_sent])
        # if config.flip_source:
        #    np_sent = np.flip(np_sent, axis=0).copy()
        tensor_sent = torch.LongTensor(np_sent)
        if self.gpu:
            tensor_sent = tensor_sent.cuda()
        src_encoding, decoder_init_state = self.encoder[key].encode_one_sent(tensor_sent)
        if config.greedy_search:
            tgt_tensor, score = self.decoder.greedy_search(
                src_encoding, decoder_init_state, max_step, replace=replace)
            tgt_np = tgt_tensor.cpu().detach().numpy()
            tgt_sent = []
            for i in tgt_np[1:-1]:
                if i >= 0:
                    tgt_sent.append(self.vocab_tgt.id2word[i])
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
                        tgt_sent.append(self.vocab_tgt.id2word[i])
                    else:
                        tgt_sent.append(src_sent[-i])
                hypotheses.append(Hypothesis(tgt_sent, score))
        return hypotheses

    def evaluate_ppl(self, dev_data, batch_size: int=32, decoder_only=False):

        cum_loss = 0.
        cum_tgt_words = 0.
        # you may want to wrap the following code using a context manager provided
        # by the NN library to signal the backend to not to keep gradient information
        # e.g., `torch.no_grad()`

        for src_sents, tgt_sents, key in batch_iter(dev_data, batch_size):
            if decoder_only:
                loss = self.decode_to_loss(tgt_sents, update_params=False)
            else:
                loss = self(src_sents, tgt_sents, key, update_params=False)
            cum_loss += loss
            tgt_word_num_to_predict = sum(len(s[1:])
                                          for s in tgt_sents)  # omitting the leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict
        if self.cum_loss_mt > 0:
            print(self.cum_loss_mt/cum_tgt_words)
            print(self.cum_loss_disc/cum_tgt_words)
            self.cum_loss_mt = 0
            self.cum_loss_disc = 0
        ppl = np.exp(cum_loss / cum_tgt_words)

        return ppl

    def step(self, loss_mt, loss_disc=None, key=None):
        # Update once every two steps to alternate languages in batches
        if loss_disc is not None:
            self.optimizer.zero_grad()
            loss_mt.backward()
            nn.utils.clip_grad_norm(self.params, config.clip_grad)
            self.optimizer.step()
        else:
            if self.train_discrim >= config.train_discriminator_for:
                self.train_discrim = 0
                self.train_not_discrim = 0
            if self.train_not_discrim >= config.train_discriminator_every:
                self.discrim_optimizer.zero_grad()
                loss_disc.backward()
                nn.utils.clip_grad_norm(self.discriminator.parameters(), config.clip_grad)
                self.discrim_optimizer.step()
                self.train_discrim += 1
            else:
                self.optimizer.zero_grad()
                loss_mt.backward()
                nn.utils.clip_grad_norm(self.params, config.clip_grad)
                self.optimizer.step()
                self.train_not_discrim += 1

    @staticmethod
    def load(model_path: str):

        model = MultiWayModel()
        print("Loading decoder")
        dec_path = model_path+".dec.pt"
        load_partial_state_dict(model.decoder, torch.load(dec_path))
        print("Loading encoders")
        for key in model.keys:
            enc_path = model_path+"."+key+".enc.pt"
            load_partial_state_dict(model.encoder[key], torch.load(enc_path))
        if model.use_discriminator:
            print("Loading discriminator")
            try:
                disc_path = model_path+".disc.pt"
                load_partial_state_dict(model.discriminator, torch.load(disc_path))
            except:
                print("Failed")

        return model

    def initialize(self):
        for k in self.keys:
            for param in self.encoder[k].parameters():
                torch.nn.init.uniform_(param, -0.1, 0.1)
        for param in self.decoder.parameters():
            torch.nn.init.uniform_(param, -0.1, 0.1)

    def load_params(self, model_path):
        if model_path is None:
            model_path = paths.model(helper=self.helper)
        dec_path = model_path + ".dec.pt"
        opt_path = model_path + ".opt.pt"
        self.optimizer.load_state_dict(torch.load(opt_path))
        self.decoder.load_state_dict(torch.load(dec_path))
        for k in self.keys:
            enc_path = model_path + "." + k + ".enc.pt"
            self.encoder[k].load_state_dict(torch.load(enc_path))
        if self.use_discriminator:
            disc_path = model_path+".disc.pt"
            self.discriminator.load_state_dict(torch.load(disc_path))

    def save(self, model_path: str):
        dec_path = model_path+".dec.pt"
        opt_path = model_path + ".opt.pt"
        torch.save(self.decoder.state_dict(), dec_path)
        torch.save(self.optimizer.state_dict(), opt_path)
        for k in self.keys:
            enc_path = model_path+"."+k+".enc.pt"
            torch.save(self.encoder[k].state_dict(), enc_path)
        if self.use_discriminator:
            disc_path = model_path+".disc.pt"
            torch.save(self.discriminator.state_dict(), disc_path)
