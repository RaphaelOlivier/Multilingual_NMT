import sentencepiece as spm

import config
import paths
import os


def train_model(lg, train_source_path, target):
    print('\nTrain subword model for ' + lg)

    vocab_size = str(config.subwords_vocab_size[lg])
    model_type = config.subwords_model_type
    print("Train subwords model")
<<<<<<< HEAD
    spm.SentencePieceTrainer.Train('--input='+train_source_path +
                                   ' --model_prefix='+target+' --model_type='+model_type+' --character_coverage=1.0 --vocab_size='+vocab_size)
=======
    spm.SentencePieceTrainer.Train('--input=' + train_source_path +
                                   ' --model_prefix=' + target + ' --character_coverage=1.0 --vocab_size=' + vocab_size)
>>>>>>> dc82d17e7f2d2ea94469b16540c785fdb43bb055


def train(lg):
    target_folder = paths.data_subwords_folder
    model_type = config.subwords_model_type
    suffix = ""
    if config.subwords_on_monolingual:
        suffix = ".with_mono"
    model_prefix = target_folder + model_type + "." + lg + suffix
    if config.subwords_on_monolingual:
        lr_path = paths.get_train_and_mono_path(lg)
    else:
        lr_path = paths.get_data_path(set="train", mode="sc", lg=lg, subwords=False, helper=False)
    train_source_path = lr_path

    train_model(lg, train_source_path, model_prefix)


class SubwordReader:
    def __init__(self, lg=None):
        if lg is None:
            lg = config.language
        folder = paths.data_subwords_folder
        model_type = config.subwords_model_type
        suffix = ""
        if config.subwords_on_monolingual:
            suffix += ".with_mono"
<<<<<<< HEAD
        model_prefix = folder+model_type+"."+lg  # + suffix
        model_path = model_prefix+".model"
=======
        model_prefix = folder + model_type + "." + lg + suffix
        model_path = model_prefix + ".model"
>>>>>>> dc82d17e7f2d2ea94469b16540c785fdb43bb055

        self.sp = spm.SentencePieceProcessor()
        print("Loading subword model :", model_path)
        self.sp.Load(model_path)

    def line_to_subwords(self, line):
        return self.sp.EncodeAsPieces(line)

    def subwords_to_line(self, l):
        return self.sp.DecodePieces(l)


def main():
    if config.subwords:
        for lg in ["az", "be", "gl", "pt", "ru", "tr"]:
            train(lg)
    else:
        print("No subwords\n")


if __name__ == '__main__':
    main()
