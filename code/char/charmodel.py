from copy import deepcopy
from nmt.nmtmodel import NMTModel
import char.charconfig as config

from utils import load_partial_state_dict
import torch


class CharModel(NMTModel):
    def __init__(self, *args, **kwargs):
        if config.language_tokens:
            super(CharModel, self).__init__(*args, helper=False, add_tokens_src=2, **kwargs)
            self.vocab.src(helper=False).add("__LOW__")
            self.vocab.src(helper=False).add("__HELPER__")
        else:
            super(CharModel, self).__init__(*args, helper=False, add_tokens_src=0, **kwargs)

    def load(model_path: str):
        enc_path = model_path+".enc.pt"
        dec_path = model_path+".dec.pt"
        model = CharModel()
        print("Loading encoder")
        load_partial_state_dict(model.encoder, torch.load(enc_path))
        print("Loading decoder")
        load_partial_state_dict(model.decoder, torch.load(dec_path))

        return model
