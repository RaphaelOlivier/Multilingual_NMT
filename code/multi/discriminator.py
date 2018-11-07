import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch


class SequenceDiscriminator(nn.Module):
    def __init__(self, sizes):
        super(SequenceDiscriminator, self).__init__()
        layers = [nn.Linear(sizes[0], sizes[1])]
        for i in range(len(sizes)-2):
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Linear(sizes[i+1], sizes[i+2]))

        self.layers = nn.Sequential(*layers)

        self.criterion = nn.BCEWithLogitsLoss(reduction="sum")

    def forward(self, contexts, labels):
        pad_contexts, lens = rnn.pad_packed_sequence(contexts)
        pad_contexts = pad_contexts.transpose(0, 1)
        piled_contexts = torch.cat([pad_contexts[i, :lens[i]] for i in range(len(lens))], dim=0)
        piled_labels = torch.cat(labels)
        reverse_labels = 1. - piled_labels
        adv_loss_mt = self.criterion(self.layers(piled_contexts).squeeze(), piled_labels)
        adv_loss_disc = self.criterion(self.layers(
            piled_contexts.detach()).squeeze(), reverse_labels)

        return adv_loss_mt, adv_loss_disc
