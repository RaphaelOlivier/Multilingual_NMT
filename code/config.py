import torch

language = "gl"

# General information
printout = True
sanity = False
load = False
pretraining = True
replace = False
seed = 1994
test = True
cuda = torch.cuda.is_available()
target_in_decode = True
use_helper = False
use_helper = use_helper and language in {"az", "be", "gl"}
# Training parameters
lr = 0.001
weight_decay = 0.0001
batch_size = 64
# clip_grad = 5.0
lr_decay = 0.2
max_epoch = 100
max_epoch_pretraining = 3
patience = 3
max_num_trial = 3
num_layers_encoder = 1
num_layers_decoder = 1
bidirectional_encoder = True
attention = True
residual = False
# Network parameters
hidden_size = 256
embed_size = 256
has_output_layer = True
dropout_layers = 0.4
#dropout_lstm_states = 0.1
# Search parameters
beam_size = 5
max_decoding_time_step = 100
greedy_search = False
# Vocab options
freq_cutoff = 2
vocab_size = 10000
vocab_mono = False
merge_target_vocab = True
# Display options
valid_niter = 200
log_every = 10


if language == "az":
    helper_language = "tr"
elif language == "gl":
    helper_language = "pt"
elif language == "be":
    helper_language = "ru"
else:
    helper_language = "None"
