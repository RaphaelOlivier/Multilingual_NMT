import torch.nn as nn
import torch
import torch.nn.utils.rnn as rnn
import numpy as np

import config
import paths
from nmt.layers import init_weights, AdvancedLSTM
from allennlp.modules.elmo import Elmo


class Encoder(nn.Module):
    def __init__(self, vocab_size):
        super(Encoder, self).__init__()
        # self.lookup = nn.Embedding(vocab_size, config.embed_size)
        options_file, weights_file = paths.get_elmo_files()
        self.lookup = Elmo(options_file, weights_file, 1, dropout=config.dropout_layers)
        self.lstm = AdvancedLSTM(config.embed_size, config.hidden_size, num_layers=config.num_layers_encoder,
                                 bidirectional=config.bidirectional_encoder, dropout=config.dropout_layers)
        self.dr = nn.Dropout(config.dropout_layers)
        self.has_output_layer = False
        self.act = nn.Tanh()
        if config.hidden_size != config.embed_size:
            self.has_output_layer = True
            self.output_layer = nn.Linear(config.hidden_size, config.embed_size)
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

        self.apply(init_weights)

    def get_word_scores(self, piled_encoded):
        h = self.dr(piled_encoded)
        if self.has_output_layer:
            h = self.act(self.output_layer(h))
        o = self.dr(h).matmul(self.lookup.weight.t())
        return o

    def forward(self, sequences, to_loss=False):
        # lens = [len(seq) for seq in sequences]
        # bounds = [0]
        # for l in lens:
        #     bounds.append(bounds[-1]+l)
        # piled_sequence = torch.cat(sequences)
        # print(piled_sequence)
        piled_sequence = sequences
        elmo_lookup = self.lookup(piled_sequence)
        embeddings, mask = elmo_lookup['elmo_representations'][0].transpose(0, 1), elmo_lookup['mask']
        lens = mask.sum(dim=1)
        embeddings = self.dr(embeddings)
        packed_sequence = rnn.pack_padded_sequence(embeddings, lens)
        encoded, last_state = self.lstm(packed_sequence)
        if not to_loss:
            return encoded, last_state
        else:
            encoded_pad, lengths = rnn.pad_packed_sequence(encoded)
            encoded_pad = encoded_pad.view(len(sequences[0]),
                                           len(sequences), 2, config.hidden_size)
            fwd = encoded_pad[:, :, 0].transpose(0, 1)
            bwd = encoded_pad[:, :, 1].transpose(0, 1)
            encoded_list = [fwd[i, :lengths[i]-1]
                            for i in range(len(lengths))] + [bwd[i, 1:lengths[i]] for i in range(len(lengths))]
            # encoded_list = [fwd[i, :lengths[i]-1] for i in range(len(lengths))]
            piled_encoded = torch.cat(encoded_list)
            scores = self.get_word_scores(piled_encoded)
            labels = torch.cat([s[1:] for s in sequences] + [s[:-1] for s in sequences])
            #labels = torch.cat([s[1:] for s in sequences])
            loss = self.criterion(scores, labels)
            return loss
            # return [scores[bounds[i]:bounds[i+1]] for i in range(len(sequences))]

    def encode_one_sent(self, seq):
        elmo_lookup = self.lookup(seq)
        embeddings, mask = elmo_lookup['elmo_representations'][0].transpose(0, 1), elmo_lookup['mask']
        encoded, last_state = self.lstm(self.dr(embeddings))
        return encoded, last_state
