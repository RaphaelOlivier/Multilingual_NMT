import io
import numpy as np
import config
import pickle

from paths import get_fasttext_path


def create_language_fasttext(language, language_vocab):
    fin = io.open(get_fasttext_path(language), 'r', encoding='utf-8', newline='\n', errors='ignore')
    embeddings = np.zeros((len(language_vocab), 300))
    n, d = map(int, fin.readline().split())
    unknowns = 0
    unknown_embedding = np.zeros(300)
    for line in fin:
        tokens = line.rstrip().split(' ')
        if language_vocab[tokens[0]] > 3:
            embeddings[language_vocab[tokens[0]]] = np.array(list(map(float, tokens[1:])))
        else:
            unknown_embedding += np.array(list(map(float, tokens[1:])))
            unknowns += 1
    embeddings[3] = unknown_embedding / unknowns
    np.save("data/monolingual/wikivecs/{}_embeddings.npy".format(language), embeddings)


def create_all_embeddings(vocab):

    print("Generating en embeddings")
    create_language_fasttext("en", vocab.tgt())
    base_languages = ["az", "be", "gl"]
    helpers = {"az": "tr", "be": "ru", "gl": "pt"}
    for language in base_languages:
        print("Generating {} embeddings".format(language))
        create_language_fasttext(language, vocab.src())
        if config.mode in ["transfer", "multi"]:
            print("Generating {} embeddings".format(helpers[language]))
            create_language_fasttext(helpers[language], vocab.src(helper=True))
