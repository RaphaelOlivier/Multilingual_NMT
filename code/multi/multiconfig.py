from config import *

embed_size_low = 256
hidden_size_encoder_low = 256
num_layers_encoder_low = 2

embed_size_helper = 256
hidden_size_encoder_helper = 256
num_layers_encoder_helper = 3

discriminator = True
smooth_disc = 0.1
adversarial_loss_coeff = 0.5
train_discriminator_every = 10
train_discriminator_for = 4
batch_size = 16
valid_niter = 700
log_every = 100
max_epoch = 10
pretraining_decoder = True
pretraining_encoders = True
max_epoch_pretraining_decoder = 2
max_epoch_pretraining_helper = 2
max_epoch_pretraining_low = 10
lr_decay = 0.2
max_num_trial = 3
patience = 3
clip_grad = 5

sampling = 4
