import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.utils.rnn as rnn
import numpy as np

import config
from nmt.layers import init_weights, AdvancedLSTM


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1,
                 bidirectional=True, dropout=0, context_projection=None, state_projection=None):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.lookup = nn.Embedding(vocab_size, self.embed_size)
        self.lstm = AdvancedLSTM(self.embed_size, self.hidden_size, num_layers=num_layers,
                                 bidirectional=bidirectional, dropout=dropout)
        self.dr = nn.Dropout(dropout)
        self.act = nn.Tanh()

        self.use_context_projection = False
        if context_projection is not None:
            self.use_context_projection = True
            self.context_projection = nn.Linear(
                self.hidden_size * (2 if bidirectional else 1), context_projection)
        self.use_state_projection = False
        if state_projection is not None:
            self.use_state_projection = True
            self.state_projection = nn.Linear(context_projection, state_projection)

        self.out_forward = nn.Linear(self.hidden_size, self.embed_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

        self.apply(init_weights)

    def forward(self, sequences, to_loss=False):
        lens = [len(seq)-1 for seq in sequences]
        bounds = [0]
        for l in lens:
            bounds.append(bounds[-1]+l)
        piled_sequence = torch.cat([s[:-1] for s in sequences])
        piled_embeddings = self.dr(self.lookup(piled_sequence))
        embed_sequences = [piled_embeddings[bounds[i]:bounds[i+1]] for i in range(len(sequences))]
        packed_sequence = rnn.pack_sequence(embed_sequences)
        encoded, last_state = self.lstm(packed_sequence)

        encoded_pad, lengths = rnn.pad_packed_sequence(encoded)

        if to_loss:
            encoded_piled = torch.cat([encoded_pad[:lens[i], i] for i in range(len(lens))])
            out = self.out_forward(encoded_piled)
            scores = F.linear(out, self.lookup.weight)
            tgt = torch.cat([s[1:] for s in sequences])
            return self.criterion(scores, tgt)
        if not self.use_context_projection:
            context_pad = encoded_pad
        else:
            context_pad = self.act(self.context_projection(self.dr(encoded_pad)))
        state = self.get_decoder_init_state(last_state)
        context = rnn.pack_padded_sequence(context_pad, lengths)

        return context, state

    def encode_one_sent(self, seq):
        embeddings = self.dr(self.lookup(seq[:-1])).unsqueeze(1)
        encoded, last_state = self.lstm(embeddings)
        if not self.use_context_projection:
            context = encoded
        else:
            context = self.act(self.context_projection(self.dr(encoded)))
        state = self.get_decoder_init_state(last_state)
        return context, state

    def get_decoder_init_state(self, state):
        if self.lstm.bidirectional:
            last_state = state[0].view(self.lstm.num_layers, 2, state[0].size(1), -1)[-1]
            last_state = torch.cat([last_state[0], last_state[1]], dim=1)
        else:
            last_state = state[0]
        if not self.use_state_projection:
            state = last_state
        else:
            state = self.act(self.state_projection(self.dr(last_state)))
        return state
