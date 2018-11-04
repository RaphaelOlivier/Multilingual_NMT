import torch.nn as nn
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

        self.apply(init_weights)

    def forward(self, sequences, to_loss=False):
        lens = [len(seq) for seq in sequences]
        bounds = [0]
        for l in lens:
            bounds.append(bounds[-1]+l)
        piled_sequence = torch.cat(sequences)
        piled_embeddings = self.dr(self.lookup(piled_sequence))
        embed_sequences = [piled_embeddings[bounds[i]:bounds[i+1]] for i in range(len(sequences))]
        packed_sequence = rnn.pack_sequence(embed_sequences)
        encoded, last_state = self.lstm(packed_sequence)

        encoded_pad, lengths = rnn.pad_packed_sequence(encoded)
        if not self.use_context_projection:
            context_pad = encoded_pad
        else:
            context_pad = self.act(self.context_projection(self.dr(encoded_pad)))
        state = self.get_decoder_init_state(context_pad)
        context = rnn.pack_padded_sequence(context_pad, lengths)

        return context, state

    def encode_one_sent(self, seq):
        embeddings = self.dr(self.lookup(seq)).unsqueeze(1)
        encoded, last_state = self.lstm(embeddings)
        if not self.use_context_projection:
            context = encoded
        else:
            context = self.act(self.context_projection(self.dr(encoded)))
        state = self.get_decoder_init_state(context)
        return context, state

    def get_decoder_init_state(self, context):
        last_state = context[0]
        if not self.use_state_projection:
            state = last_state
        else:
            state = self.act(self.state_projection(self.dr(last_state)))

        return state
