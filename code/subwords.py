import sentencepiece as spm

import config
import paths
import os


def train_model(lg, train_source_path, target):
    print('\nTrain subword model for '+lg)

    vocab_size = str(config.subwords_vocab_size[lg])
    print("Train subwords model")
    spm.SentencePieceTrainer.Train('--input='+train_source_path +
                                   ' --model_prefix='+target+' --character_coverage=1.0 --vocab_size='+vocab_size)


def train(lg, helper):
    target_folder = paths.data_subwords_folder
    model_type = config.subwords_model_type
    suffix = ".with_helper" if helper else ".no_helper"
    model_prefix = target_folder+model_type+"."+lg+suffix

    if not helper:
        train_source_path = paths.get_data_path(
            set="train", mode="sc", lg=lg, subwords=False, helper=False)
    else:
        print("Generate temp file")
        temp_path = paths.data_bilingual_folder+"temp.txt"
        lr_file = paths.get_data_path(set="train", mode="sc", lg=lg, subwords=False, helper=False)
        hr_file = paths.get_data_path(set="train", mode="sc", lg=lg, subwords=False, helper=True)
        with open(temp_path, 'w') as f, open(lr_file, 'r') as s1, open(hr_file, 'r') as s2:
            for line in s1:
                f.write(line)
            for line in s2:
                f.write(line)

        train_source_path = temp_path

    train_model(lg, train_source_path, model_prefix)

    if helper:
        os.remove(temp_path)
        print("Temp file removed")


class SubwordReader:
    def __init__(self, lg):
        if lg is None:
            lg = config.language
        folder = paths.data_subwords_folder
        model_type = config.subwords_model_type
        helper = config.use_helper or config.mode != "normal"
        suffix = ".with_helper" if helper else ".no_helper"
        model_prefix = folder+model_type+"."+lg+suffix
        model_path = model_prefix+".model"
        sp = spm.SentencePieceProcessor()
        sp.Load(model_path)

    def line_to_subwords(line):
        return sp.EncodeAsPieces(line)

    def subwords_to_line(l):
        return sp.DecodePieces(l)


def main():
    if config.subwords:
        helper = config.use_helper or config.mode != "normal"
        for lg in ["az", "be", "gl"]:
            train(lg, helper)
    else:
        print("No subwords\n")


if __name__ == '__main__':
    main()
