import math
from typing import List
from collections import namedtuple
import config
import numpy as np
from torch.nn import Parameter
import subwords

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


def zip_data(*args):
    assert len(args) >= 2, "Missing an argument in zip_data"
    if len(args) == 2:
        src = args[0]
        tgt = args[1]
        assert len(src) == len(tgt)
        key = None
        return {key: list(zip(src, tgt))}

    assert len(args) % 3 == 0, "Missing an argument in zip_data"
    d = {}
    for i in range(len(args)//3):
        src = args[3*i]
        tgt = args[3*i+1]
        key = args[3*i+2]
        assert key not in d.keys(), "Twice the same key in zip_data"
        d[key] = list(zip(src, tgt))
    return d


def input_transpose(sents, pad_token):
    """
    This function transforms a list of sentences of shape (batch_size, token_num) into
    a list of shape (token_num, batch_size). You may find this function useful if you
    use pytorch
    """
    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    sents_t = []
    for i in range(max_len):
        sents_t.append([sents[k][i] if len(sents[k]) > i else pad_token for k in range(batch_size)])

    return sents_t


def read_corpus(file_path, source='src', lg=None):
    if config.subwords and source == 'src':
        sub = subwords.SubwordReader(lg)
    print(file_path)
    test = "test" in file_path
    data = []
    counter = 0
    for line in open(file_path):
        if config.subwords and source == 'src':
            sent = sub.line_to_subwords(line)
            # print(sent)
        else:
            sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == "tgt":
            sent = ['<s>'] + sent + ['</s>']
        if len(sent) <= config.max_len_corpus or test:
            data.append(sent)
        else:
            counter += 1
    print("Eliminated :", counter, "out of", len(data))

    return data


def write_sents(sents, path):
    # if config.subwords:
    #    sub = subwords.SubwordReader()
    with open(path, 'w') as f:
        for sent in sents:
            # if config.subwords:
            #    line = sub.subwords_to_line(sent)
            # else:
            #    line = ' '.join(sent)
            line = ' '.join(sent)
            f.write(line + '\n')


def batch_iter_one_way(data, batch_size, shuffle=False, key=None):
    """
    Given a list of examples, shuffle and slice them into mini-batches
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents, key


def batch_iter(data, batch_size, shuffle=False, sampling_multi=1):

    if len(data.keys()) == 1:
        for t in batch_iter_one_way(list(data.values())[0], batch_size, shuffle=shuffle, key=list(data.keys())[0]):
            yield t

    else:
        keys = list(data.keys())
        lens = np.array([len(data[k]) for k in keys])
        max_key = keys[np.argmax(lens)]
        multi_batch_num = batch_num = math.ceil(len(data[max_key]) / batch_size)
        index_arrays = [list(range(l)) for l in lens]
        if shuffle:
            for a in index_arrays:
                np.random.shuffle(a)
        current_index = [0 for k in keys]

        keys_indexes = range(len(keys))
        if len(keys) == 2:
            max_key_id = np.argmax(lens)
            min_key_id = 1 - max_key_id
            keys_indexes = [min_key_id] + sampling_multi*[max_key_id]

        for j in range(0, batch_num, sampling_multi):
            for i in keys_indexes:
                k = keys[i]
                if current_index[i] >= lens[i]:
                    current_index[i] = 0
                    if shuffle:
                        np.random.shuffle(index_arrays[i])
                current_index[i] += batch_size
                indices = index_arrays[i][current_index[i]-batch_size:current_index[i]]

                examples = [data[k][idx] for idx in indices]

                examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
                src_sents = [e[0] for e in examples]
                tgt_sents = [e[1] for e in examples]

                yield src_sents, tgt_sents, k


def load_partial_state_dict(model, state_dict):

    model_state = model.state_dict()
    loaded_keys = []
    unloaded_keys = []
    unseen_keys = []
    for name, param in state_dict.items():
        if name not in model_state:
            unloaded_keys.append(name)
            continue
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        model_state[name].copy_(param)
        loaded_keys.append(name)
    for name, param in model_state.items():
        if name not in loaded_keys:
            unseen_keys.append(name)
    if len(unseen_keys) > 0:
        print("Some params not found in file :", unseen_keys)
    if len(unloaded_keys) > 0:
        print("Some params in file not in model :", unloaded_keys)
    if len(unseen_keys) == 0 and len(unloaded_keys) == 0:
        print("Model and file matching !")
