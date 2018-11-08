import torch.nn as nn
import torch
import torch.nn.utils.rnn as rnn
import numpy as np

import config
from nmt.layers import MultipleLSTMCells, init_weights


class Decoder(nn.Module):

    def __init__(self, vocab_size):
        super(Decoder, self).__init__()
        self.act = nn.Tanh()
        self.hidden_size = config.hidden_size * (2 if config.bidirectional_encoder else 1)
        self.lookup = nn.Embedding(vocab_size, config.embed_size)
        self.nlayers = config.num_layers_decoder
        self.tie_embeddings = True
        self.dr = nn.Dropout(config.dropout_layers)
        self.final_layers_input = self.hidden_size
        self.input_lstm_size = config.embed_size
        self.attention = False
        if config.attention:
            self.attention = True
            self.input_lstm_size = config.embed_size + self.hidden_size
            self.att_layer = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.final_layers_input = self.hidden_size + self.hidden_size
        lstm_sizes = [self.input_lstm_size] + [self.hidden_size for i in range(self.nlayers)]
        self.lstm = MultipleLSTMCells(self.nlayers, lstm_sizes, residual=config.residual,
                                      dropout_vertical=config.dropout_layers, dropout_horizontal=config.dropout_lstm_states)

        self.score_input = self.final_layers_input
        self.has_output_layer = config.has_output_layer or (self.score_input != config.embed_size)
        if self.has_output_layer:
            self.score_input = config.embed_size
            self.output_layer = nn.Linear(self.final_layers_input, config.embed_size)

        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.init_context = torch.autograd.Variable(torch.Tensor(np.zeros(self.hidden_size))).cuda()

        self.apply(init_weights)

    def get_params_but_embedding(self):
        return list(self.att_layer.parameters())+list(self.lstm.parameters())+(list(self.output_layer.parameters()) if self.has_output_layer else [])

    def merge_directions(self, h):
        if config.bidirectional_encoder:
            sep = h.view(config.num_layers_decoder, 2, -1, config.hidden_size)
            forw = sep[:, 0]
            bac = sep[:, 1]
            bi = torch.cat([forw, bac], dim=2)
            return bi
        else:
            return h

    def get_word_scores(self, decs, contexts):
        if self.attention:
            outs = torch.cat([decs, contexts], dim=1)
        else:
            outs = decs
        h = outs
        if self.has_output_layer:
            h = self.act(self.output_layer(h))
        if self.tie_embeddings:
            return self.dr(h).matmul(self.lookup.weight.t())
        else:
            return self.score_layer(h)

    def get_init_state(self, encoder_state, keep_all=True):
        if keep_all:
            init_state = encoder_state
        else:
            init_state = encoder_state[0].new_full(
                encoder_state[0].size(), 0), encoder_state[1].new_full(encoder_state[1].size(), 0)
            init_state[0][0] = encoder_state[0][-1]
            init_state[1][0] = encoder_state[1][-1]

        return init_state

    def one_step_decoding(self, input, state, encoded, attention_mask=None, attend=True, replace=False):

        o = input
        h, new_state = self.lstm(o, (state[0], state[1]))
        o = self.dr(h)
        # attention
        context = None
        if self.attention and attend:
            att_scores = torch.bmm(encoded, o.unsqueeze(2))
            if len(att_scores[0]) == 11:
                # print(attention_mask.cpu().detach().numpy())
                # print(att_scores.cpu().detach().numpy())
                pass
            if attention_mask is not None:
                att_scores.masked_fill_(attention_mask, float("-inf"))
            if len(att_scores[0]) == 11:
                # print(att_scores.cpu().detach().numpy())
                pass

            att_scores = nn.Softmax(dim=1)(att_scores)
            if len(att_scores[0]) == 11:
                # print(att_scores.cpu().detach().numpy())
                pass

            context = torch.bmm(encoded.transpose(1, 2), att_scores).squeeze(2)
            val_sort, arg_sort = torch.sort(att_scores, dim=1, descending=True)
        else:
            context = o.new_full(o.size(), 0)

        if replace:
            return o, context, new_state, arg_sort[0][0]
        else:
            return o, context, new_state

    def forward(self, encoded, encoder_state, tgt_sequences, attend=True):
        # Embedding

        lens = [len(seq) for seq in tgt_sequences]
        seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
        sorted_lens = [lens[i] for i in seq_order]
        inv_sort = sorted(range(len(seq_order)), key=seq_order.__getitem__, reverse=False)
        bounds = [0]
        for l in lens:
            bounds.append(bounds[-1] + l)
        piled_sequence = torch.cat(tgt_sequences)
        piled_embeddings = self.dr(self.lookup(piled_sequence))
        embed_sequences = [piled_embeddings[bounds[i]:bounds[i + 1]]
                           for i in range(len(tgt_sequences))]
        # sorting target sequences
        sorted_embed_sequences = [embed_sequences[i] for i in seq_order]
        sorted_bounds = [0]
        for i in seq_order:
            sorted_bounds.append(sorted_bounds[-1] + lens[i])
        padded_tgt_sequence = rnn.pad_sequence(sorted_embed_sequences)

        # sorting encoder outputs
        if attend:
            sorted_encoder_state = encoder_state[0][:,
                                                    seq_order, :], encoder_state[1][:, seq_order, :]
            sorted_encoder_state = self.merge_directions(
                sorted_encoder_state[0]), self.merge_directions(sorted_encoder_state[1])
            padded_enc, lengths_enc = rnn.pad_packed_sequence(encoded)
            sorted_encoder_outputs = padded_enc.transpose(0, 1)[seq_order]
            sorted_enc_lengths = [lengths_enc[i] for i in seq_order]
            sorted_attention_mask = sorted_encoder_outputs.new_full(
                (sorted_encoder_outputs.size(0), sorted_encoder_outputs.size(1), 1), 0, dtype=torch.uint8)

            for i in range(len(sorted_enc_lengths)):
                sorted_attention_mask[i, sorted_enc_lengths[i]:] = 1
            state = self.get_init_state(sorted_encoder_state)
        else:
            sorted_attention_mask = None
            sorted_encoder_outputs = None
            state = piled_embeddings.new_full((self.nlayers, len(tgt_sequences), self.hidden_size),
                                              0), piled_embeddings.new_full(
                (self.nlayers, len(tgt_sequences), self.hidden_size), 0)

        output_list = []
        prev_context_vector = self.init_context.expand(state[0][-1].size())
        if self.attention:
            context_list = []
        if config.residual:
            self.lstm.sample_masks()
        for i in range(len(padded_tgt_sequence)):
            if self.attention:
                input = torch.cat([padded_tgt_sequence[i], prev_context_vector], dim=1)
            else:
                input = padded_tgt_sequence[i]
            output, new_context_vector, state = self.one_step_decoding(
                input, state, sorted_encoder_outputs, sorted_attention_mask, attend=attend)
            output_list.append(output.unsqueeze(0))
            if self.attention:
                context_list.append(new_context_vector.unsqueeze(0))
                prev_context_vector = new_context_vector

        padded_dec = torch.cat(output_list, dim=0)
        list_dec = [padded_dec.transpose(0, 1)[i, :sorted_lens[i] - 1]
                    for i in range(len(tgt_sequences))]
        piled_dec = torch.cat(list_dec)
        if self.attention:
            padded_contexts = torch.cat(context_list, dim=0)
            list_contexts = [padded_contexts.transpose(0, 1)[i, :sorted_lens[i] - 1]
                             for i in range(len(tgt_sequences))]
            piled_contexts = torch.cat(list_contexts)
        # Attention
        else:
            piled_contexts = None

        # Scoring
        piled_scores = self.get_word_scores(piled_dec, piled_contexts)
        sorted_targets = [tgt_sequences[i][1:] for i in seq_order]
        labels = torch.cat(sorted_targets)
        loss = self.criterion(piled_scores, labels)
        return loss
        """
        sorted_scores = [list_scores[sorted_bounds[i]:sorted_bounds[i+1]]
                         for i in range(len(tgt_sequences))]
        scores = [sorted_scores[i] for i in inv_sort]
        return scores
        """

    def greedy_search(self, encoded_sequence, init_state, max_step=None, replace=False):
        self.eval()
        if max_step is None:
            max_step = config.max_decoding_time_step
        tgt_ids = []
        start = torch.LongTensor([1])
        stop = torch.LongTensor([2])
        score = torch.FloatTensor([0])
        if self.cuda:
            start = start.cuda()
            stop = stop.cuda()
            score = score.cuda()
        current_word = start
        tgt_ids.append(current_word)
        time_step = 0
        state = self.merge_directions(init_state[0]), self.merge_directions(init_state[1])
        prev_context_vector = self.init_context.expand(state[0][-1].size())
        encoder_outputs = encoded_sequence.transpose(0, 1)
        while (time_step < max_step and current_word[0] != stop[0]):
            time_step += 1
            embed = self.dr(self.lookup(current_word))
            if self.attention:
                input = torch.cat([embed, prev_context_vector], dim=1)
            else:
                input = embed
            if replace:
                dec, new_context_vector, state, attended = self.one_step_decoding(
                    input, state, encoder_outputs, replace=replace)
            else:
                dec, new_context_vector, state = self.one_step_decoding(
                    input, state, encoder_outputs, replace=replace)

            prev_context_vector = new_context_vector

            scores = self.get_word_scores(dec, new_context_vector)
            current_score, current_word = torch.max(scores, dim=1)

            if replace and self.attention and current_word.item() == 3:
                # make sure most attended word isn't unk
                translated_word = -attended
            else:
                translated_word = current_word

            tgt_ids.append(translated_word)
            score += current_score

        if current_word != stop:
            tgt_ids.append(stop)
        score = (score.cpu().detach().numpy() - np.log(time_step)).item()
        return torch.cat(tgt_ids), score

    def beam_search(self, encoded_sequence, init_state, beam_size, max_step=None, replace=False):

        if max_step is None:
            max_step = config.max_decoding_time_step
        if beam_size is None:
            beam_size = config.beam_size
        start = torch.LongTensor([1])
        stop = torch.LongTensor([2])
        if self.cuda:
            start = start.cuda()
            stop = stop.cuda()
        state = self.merge_directions(init_state[0]), self.merge_directions(init_state[1])
        time_step = 0
        prev_context_vector = self.init_context.expand(state[0][-1].size())
        encoder_outputs = encoded_sequence.transpose(0, 1)
        hypotheses = [([start], state, prev_context_vector, 0)]

        while (time_step < max_step):

            time_step += 1
            next_hypotheses = []
            stopped = True

            for hypothesis in hypotheses:
                current_sentence = hypothesis[0]
                if current_sentence[-1] >= 0:
                    current_word = current_sentence[-1]
                else:
                    current_word = torch.cuda.LongTensor([3])
                if current_word == stop:
                    next_hypotheses.append(hypothesis)
                    continue
                else:
                    stopped = False
                current_state = hypothesis[1]
                current_score = hypothesis[3]

                embed = self.lookup(current_word)
                if self.attention:
                    input = torch.cat([embed, hypothesis[2]], dim=1)
                else:
                    input = embed
                if replace:
                    dec, new_context_vector, current_state, attended = self.one_step_decoding(
                        input, current_state, encoder_outputs, replace=replace)
                else:
                    dec, new_context_vector, current_state = self.one_step_decoding(
                        input, current_state, encoder_outputs, replace=replace)

                scores = self.get_word_scores(dec, new_context_vector)

                probs = -nn.LogSoftmax(dim=1)(scores).squeeze()
                probs = probs.detach().cpu().numpy()
                max_indices = np.argpartition(probs, beam_size - 1)[:beam_size]
                if replace and self.attention:
                    max_indices[max_indices == 3] = -attended.item()
                next_hypothesis = []
                for i in max_indices:
                    if i < 0:
                        prob_idx = 3
                    else:
                        prob_idx = i

                    next_hypothesis.append(
                        (
                            current_sentence + [torch.cuda.LongTensor([i])],
                            current_state,
                            new_context_vector,
                            score_update(current_score, probs[prob_idx], time_step)
                        )
                    )
                next_hypotheses.extend(next_hypothesis)
            if stopped:
                break

            beam_scores = np.array([hypothesis[3] for hypothesis in next_hypotheses])
            beam_indices = np.argpartition(beam_scores, beam_size - 1)[:beam_size]
            hypotheses = [next_hypotheses[j] for j in beam_indices]

        return sorted([(torch.cat(hypothesis[0]), hypothesis[3]) for hypothesis in hypotheses], key=lambda x: x[1])


def score_update(old_score, update, time_step):

    return old_score * (time_step - 1) / time_step + update / time_step
