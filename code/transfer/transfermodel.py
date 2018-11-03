from copy import deepcopy
import torch.nn as nn
import torch

import transfer.transferconfig as config
from nmt.nmtmodel import NMTModel


class TransferModel(NMTModel):
    def __init__(self):
        super(TransferModel, self).__init__(helper=True)
        self.saved_helper_embeddings = None
        self.saved_main_embeddings = None
        self.optimizer_main = None

    def switch(self):
        get_to_gpu = False
        if self.gpu:
            self.to_cpu()
            get_to_gpu = True
        if self.helper:
            self.saved_helper_embeddings = deepcopy(self.encoder.lookup)
            if self.saved_main_embeddings is None:
                self.encoder.lookup = nn.Embedding(
                    len(self.vocab.src(helper=False)), config.embed_size)
                weights_indices = torch.randperm(len(self.vocab.src(helper=False)))
                weights = self.saved_helper_embeddings.weight[weights_indices]
            else:
                self.encoder.lookup = deepcopy(self.saved_main_embeddings)
            self.params = self.get_main_params()
            self.optimizer = torch.optim.Adam(
                self.params, lr=config.lr, weight_decay=config.weight_decay)
            self.helper = False
        else:
            self.saved_main_embeddings = deepcopy(self.encoder.lookup)
            self.params = list(self.encoder.parameters())+list(self.decoder.parameters())
            self.optimizer = torch.optim.Adam(
                self.params, lr=config.lr, weight_decay=config.weight_decay)
            self.helper = True
        if get_to_gpu:
            self.to_gpu()

    def get_main_params(self):
        if config.transfer_freeze == "decoder":
            return list(self.encoder.parameters())
        elif config.transfer_freeze == "decoder_embeddings":
            return list(self.encoder.parameters()) + self.decoder.get_params_but_embedding()
        elif config.transfer_freeze == "all_but_encoder_embedding":
            return list(self.encoder.lookup.parameters())
        else:
            assert config.transfer_freeze == "none"
            return list(self.encoder.parameters())+list(self.decoder.parameters())

    @staticmethod
    def load(model_path: str, helper=False):
        """
        Load a pre-trained model

        Returns:
            model: the loaded model
        """
        enc_path = model_path+".enc.pt"
        dec_path = model_path+".dec.pt"
        model = TransferModel()  # initialization is in the helper domain
        if not helper:
            print("Switching in loading")
            model.switch()
        model.encoder.load_state_dict(torch.load(enc_path))
        model.decoder.load_state_dict(torch.load(dec_path))

        return model
