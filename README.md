# Low-resource translation
Modular set of models for multilingual translationapplied to low-resource languages

## Organization
* `decoded_results` : our decoding results on the validation and test sets
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
    
  

## Usage  
### Prerequisites :
* numpy>=1.15.1
* pytorch 0.4.1
* tqlm
* docopt
* nltk
* sentencepiece

### Before running
To use that code, you should have an additional `data` folder in the root, with :
    * the WMT 2015 data files for Azeri, Belarusian, Galician, Turkish, Russian and Portuguese, in `data/bilingual`
    * (unused) wikipedia data for Azeri, Belarusian, Galician in `data/monolingual`
    * initially empty `data/vocab` and `data/subwords` folders.

### Run the code
All scripts are run from the main directory.
Before the first launch, run `python code/subwords.py`, then `python code/vocab.py`.
Models can be trained with `./scripts/train.sh` amd run with `./scripts/decode.sh`.
Every option is decided by editing the `config.py` file, or if specific to a mode the other config files.

### Play with options
Here are the main options to be changed
* Change the low-resource language used with `config.language` (az, be or gl)
* Decode on test file with `config.test = True` (otherwise dev file)
* Change mode with `config.mode` between `normal`, `shared`, `transfer`, or `multi`. `shared` mode correspond to alternate minibatch sampling.
* Activate or deactivate subwords with `config.subwords`
* Deactivate the use of helper language with `config.use_helper = False`. Only for normal mode. `True` actually correspond to shared encoder multitask with concatenated corpuses.


