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
all_languages = ["az", "be", "ru", "gl", "pt", "tr"]

# General information
printout = True
sanity = False
load = False
pretraining = False
pretraining_encoder = False
replace = True
seed = 1999
test = True
cuda = torch.cuda.is_available()
target_in_decode = True

# Helper language settings
use_helper = True
# flip_source = False  # keep at false
use_helper = use_helper and language in {"az", "be", "gl"}
start_ratio = 10
end_ratio = 3

# General training parameters
lr = 0.0001
weight_decay = 0.00001
batch_size = 128
mono_batch_size = 8
# clip_grad = 5.0
lr_decay = 0.2
max_epoch = 10
max_epoch_pretraining = 1
max_epoch_pretraining_encoder = 1
patience = 3
max_num_trial = 3

# Network parameters
num_layers_encoder = 2
num_layers_decoder = 2
bidirectional_encoder = True
residual = True
hidden_size_encoder = 256
hidden_size_decoder = 512
embed_size = 256
has_output_layer = True
dropout_layers = 0.5
dropout_lstm_states = 0.2
context_size = 256

# Search parameters
beam_size = 5
max_decoding_time_step = 100
greedy_search = False

# Vocab options
freq_cutoff = 2
vocab_size = 50000
vocab_mono = False
max_len_corpus = 1000
merge_target_vocab = False
merge_lr_and_helper_vocab = True

# Display options
valid_niter = 500
log_every = 50

# Low resource options
mode = "char"
