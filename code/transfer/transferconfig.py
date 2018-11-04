from config import *
transfer_freeze = "decoder_embeddings"
load_helper_model = False
train_helper_model = True
decode_helper_model = False


def max_epoch(helper): return 20 if helper else 200


def valid_niter(helper): return 3000 if helper else 500


def log_every(helper): return 300 if helper else 50
