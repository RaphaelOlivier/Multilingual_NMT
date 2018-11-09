from config import *

# bilingual to single training parameters
use_helper = True
use_helper = use_helper and language in {"az", "be", "gl"}

# sampling ratios for HRL : LRL
# -1 indicates using all of HRL
start_ratio = -1
end_ratio = 3

pretraining = False
pretraining_encoder = False

max_len_corpus *= 5
