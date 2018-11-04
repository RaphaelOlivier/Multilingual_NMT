from config import *

embed_size_low = 256
hidden_size_encoder_low = 256
num_layers_encoder_low = 1

embed_size_helper = 256
hidden_size_encoder_helper = 256
num_layers_encoder_helper = 3

batch_size = 16
valid_niter = 2000
log_every = 100
max_epoch = 10
pretraining_decoder = True
pretraining_encoders = True
max_epoch_pretraining_decoder = 1
max_epoch_pretraining_helper = 1
max_epoch_pretraining_low = 5
lr_decay = 0.5
max_num_trial = 10
patience = 2
