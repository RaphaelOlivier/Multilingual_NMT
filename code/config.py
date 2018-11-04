import torch

language = "az"

# Helper language parameters
if language == "az":
    helper_language = "tr"
elif language == "be":
    helper_language = "ru"
elif language == "gl":
    helper_language = "pt"
else:
    helper_language = "None"

# General information
printout = True
sanity = False
load = True
pretraining = True
replace = True
seed = 1995
test = True
cuda = torch.cuda.is_available()
target_in_decode = True
use_helper = False
# flip_source = False  # keep at false
use_helper = use_helper and language in {"az", "be", "gl"}
# General training parameters
lr = 0.001
weight_decay = 0.00001
batch_size = 16
# clip_grad = 5.0
lr_decay = 0.2
max_epoch = 200
max_epoch_pretraining = 4
patience = 3
max_num_trial = 3
# Network parameters
num_layers_encoder = 2
num_layers_decoder = 2
bidirectional_encoder = True
residual = False
hidden_size_encoder = 256
hidden_size_decoder = 512
embed_size = 256
has_output_layer = True
dropout_layers = 0.4
dropout_lstm_states = 0.2
context_size = 256
# Search parameters
beam_size = 5
max_decoding_time_step = 100
greedy_search = False
# Vocab options
freq_cutoff = 2
vocab_size = 15000
vocab_mono = False
merge_target_vocab = True
# Display options
valid_niter = 500
log_every = 50

# Low resource options
mode = "multi"
