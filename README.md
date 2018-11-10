# Low-resource translation
Modular set of models for multilingual translationapplied to low-resource languages

## Organization
* `results` : our decoding results on the validation and test sets
* `scripts` : training and decoding scripts
* `code` : All pytohn code
    * `paths.py` : paths to all data files
    * `config.y` : where all hyperparameters and other options are stored.
    * `utils.py` : auxiliary function for data reading and iterations
    * `subwords.py` : word segmentation training and decoding
    * `vocab.py` : vocabulary extraction file
    * `extract_ted_talks.py` : data extraction file from the provided starter code
    * `run.py` : main file called by the scripts
    * `nmt` : the baseline model and main network components, used in all other models
        * `layers.py` : enhanced version of pytorch's LSTM, LSTMCell and Dropout modules.
        * `routine.py` : main training, pretraining, searching and scoring functions for any model and parameters
        * `encoder.py` : the encoder module
        * `decoder.py` : the decoder module
        * `nmtmodel.py` : a simple sequence-to-sequence model, designed for main_training
        * `nmt.py` : data loading, model loading or initialization, and decoding functions
    * `multi`, `shared` and `transfer` : other models with their own versions of the `nmt.py`, `nmtmodel.py` and `config.py` files. They correspond respectively to multiple-encoder multitask, shared-encoder multitask, and transfer learning approaches.
    
    
