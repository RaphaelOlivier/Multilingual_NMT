from copy import deepcopy
from nmt.nmtmodel import NMTModel
import shared.sharedconfig as config
from utils import load_partial_state_dict
import torch


class SharedModel(NMTModel):

    def __init__(self, *args, **kwargs):
        if config.start_language_token:
            add_tokens_src = 2
        else:
            add_tokens_src = 0
        super(SharedModel, self).__init__(*args, helper=False, add_tokens_src=add_tokens_src, **kwargs)
        if config.start_language_token:
            self.vocab.src(False).add("__LOW__")
            self.vocab.src(False).add("__HELPER__")

    def load(model_path: str):

        enc_path = model_path + ".enc.pt"
        dec_path = model_path + ".dec.pt"
        model = SharedModel()
        print("Loading encoder")
        dict = torch.load(enc_path)
        for name, params in dict.items():
            print(name)
            print(params.shape)
        load_partial_state_dict(model.encoder, torch.load(enc_path))
        print("Loading decoder")
        load_partial_state_dict(model.decoder, torch.load(dec_path))

        return model

    def __call__(self, src_sents, tgt_sents, key=None, **kwargs):
        if key == "low":
            src_sents2 = [["__LOW__"] + s for s in src_sents]
        else:
            assert key == "helper", "wrong key : " + str(key)
            src_sents2 = [["__HELPER__"] + s for s in src_sents]
        if config.start_language_token:
            return super(SharedModel, self).__call__(src_sents2, tgt_sents, key=key, **kwargs)
        return super(SharedModel, self).__call__(src_sents, tgt_sents, key=key, **kwargs)

    def encode_to_loss(self, src_sents, key=None, **kwargs):
        if key == "low":
            src_sents2 = [["__LOW__"] + s for s in src_sents]
        else:
            assert key == "helper", "wrong key : " + str(key)
            src_sents2 = [["__HELPER__"] + s for s in src_sents]
        if config.start_language_token:
            return super(SharedModel, self).encode_to_loss(src_sents2, key=key, **kwargs)
        return super(SharedModel, self).encode_to_loss(src_sents, key=key, **kwargs)

    def beam_search(self, src_sent, key=None, **kwargs):
        if key == "low":
            src_sent2 = ["__LOW__"] + src_sent
        else:
            assert key == "helper", "wrong key : " + str(key)
            src_sent2 = ["__HELPER__"] + src_sent
        if config.start_language_token:
            return super(SharedModel, self).beam_search(src_sent2, key=key, **kwargs)
        return super(SharedModel, self).beam_search(src_sent, key=key, **kwargs)
