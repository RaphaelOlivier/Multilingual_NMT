import os
import json
from collections import Counter
from contextlib import ExitStack
import random

import paths
from embeddings import elmo_config

random.seed(elmo_config.random_seed)


def unked_output(l):
    unked_list = ['<UNK>' if vocab_counts[word] == 1 else word for word in l.split(' ')]
    output = '<S> ' + ' '.join(unked_list) + ' </S>'

    return output


if __name__ == '__main__':

    for language in 'az', 'be', 'gl':

        language_dir = os.path.join(paths.elmo_directory, language)
        train_dir = os.path.join(language_dir, 'training_data')
        heldout_dir = os.path.join(language_dir, 'heldout_data')
        save_dir = os.path.join(language_dir, 'save_dir')

        input_file = paths.get_mono_path(language)
        vocab_counts = {}
        line_count = 0

        for l in open(input_file, 'r'):

            l = l[:-1]
            line_count += 1
            for word in l.split(" "):
                try:
                    vocab_counts[word] += 1
                except KeyError:
                    vocab_counts[word] = 1

        training_filenames = [os.path.join(train_dir, 'training{}.txt'.format(i)) for i in
                              range(elmo_config.training_file_number)]
        heldout_filenames = [os.path.join(heldout_dir, 'heldout{}.txt'.format(i)) for i in
                             range(elmo_config.heldout_file_number)]

        with ExitStack() as stack:
            training_files = [stack.enter_context(open(fname, 'w')) for fname in training_filenames]
            heldout_files = [stack.enter_context(open(fname, 'w')) for fname in heldout_filenames]

            for i, l in enumerate(open(input_file, 'r')):

                l = l[:-1]
                if random.random() < elmo_config.heldout_ratio:
                    filename_index = random.randrange(elmo_config.heldout_file_number)
                    heldout_files[filename_index].write(unked_output(l) + '\n')
                else:
                    filename_index = random.randrange(elmo_config.training_file_number)
                    training_files[filename_index].write(unked_output(l) + '\n')

        ordered_vocab = sorted(vocab_counts.keys(), key=lambda word: vocab_counts[word], reverse=True)

        with open(os.path.join(language_dir, 'vocab.txt'), 'w') as f:

            f.write('<S>\n</S>\n<UNK>')

            for word in ordered_vocab:

                f.write('\n' + word)
