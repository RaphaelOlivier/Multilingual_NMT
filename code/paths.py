import config


results_folder = "results/"


def model(helper): return results_folder+"model." + \
    (config.helper_language if helper else config.language) + ".bin"


def get_vocab_path(l): return "data/vocab/"+l+".bin"


vocab = get_vocab_path(config.language)
vocab = "data/vocab/all.bin"
data_bilingual_folder = "data/bilingual/"
data_subwords_folder = "data/subwords/"

decode_output_suffix = ".test.txt" if config.test else ".valid.txt"
decode_output = results_folder+"decode."+config.language+decode_output_suffix


def get_data_path(set, mode, helper=False, lg=None, subwords=False):
    if subwords:
        prefix = data_subwords_folder
    else:
        prefix = data_bilingual_folder
    if lg == None:
        lg = config.language
    if helper and lg in ["az", "be", "gl"]:
        lg = config.get_helper_language(lg)
    if mode == "tg":
        suffix = ".en.txt"
    else:
        assert mode == "sc"
        suffix = "."+lg+".txt"
    return prefix+set + ".en-"+lg+suffix


def get_mono_path(lg=None):
    if lg == None:
        lg = config.language
    return "data/monolingual/"+lg+".wiki.txt"


train_source = get_data_path("train", "sc", helper=False)
train_target = get_data_path("train", "tg", helper=False)
train_source_helper = get_data_path("train", "sc", helper=True)
train_target_helper = get_data_path("train", "tg", helper=True)
dev_source = get_data_path("dev", "sc", helper=False)
dev_target = get_data_path("dev", "tg", helper=False)
dev_source_helper = get_data_path("dev", "sc", helper=True)
dev_target_helper = get_data_path("dev", "tg", helper=True)
test_source = get_data_path("test", "sc", helper=False)
test_target = get_data_path("test", "tg", helper=False)
test_source_helper = get_data_path("test", "sc", helper=True)
test_target_helper = get_data_path("test", "tg", helper=True)

data_monolingual = "data/monolingual/"+config.language+".wiki.txt"
