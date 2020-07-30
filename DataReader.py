import numpy as np
import os
import sys
import pickle
import re
import tqdm
import logging  # thulac,logging
from functools import partial
from collections import Counter
from random import shuffle, randint
MARK_PAD = "<PAD>"
MARK_UNK = "<UNK>"


def raw_data_import(path="./data",
                    filenames=["离婚_parsed_0_50000.dat",
                               "离婚_parsed_50000_100000.dat", "离婚_parsed_100000_122937.dat"],
                    segmentation_tool="./THULAC-Python-master/",
                    Law_text="./data/婚姻法.txt",
                    Text_path="./data/Text_all.txt",
                    all_petition=False):  
    raw_data = []
    for file in filenames:
        with open(os.path.join(path, file), "rb") as f:
            raw_data += pickle.load(f)
    logging.info("Raw data loaded. %d cases in total." % len(raw_data))
    data = []
    Labels = {"准许": 1, "驳回": 0}
    with open(Law_text, "r") as f:
        Law = f.read().replace("\n", "")
    if not os.path.isfile(Text_path):
        Corpus = Law + "。".join([case[0][0].replace("&", "") + "。".join(["。".join([i for i in c if i]) for c in case[4]]) for case in raw_data])
        Corpus = Corpus.replace("\n", "")
        with open(Text_path, "w") as f:
            f.write(Corpus)
        del Corpus
    yield Text_path  # corpus's location, used to build dicitonary and train word_vector
    yield None, None  # seg(Law).split(),Law

    for case in raw_data:
        p = re.sub("&*", "", case[0][0])
        try:
            if not all_petition:
                q = [list(filter(lambda x:re.search("(^[^，。]*离婚\S*$)|(^[^，。]*解除[^，。]*?婚姻\S*$)", x[0]), case[4]))[0]]
                labels = [Labels[q[0][2]]]
                q = [(q[0][0], q[0][3])]
            else:
                q = [list(filter(lambda x:re.search("(^[^，。]*离婚\S*$)|(^[^，。]*解除[^，。]*?婚姻\S*$)", x[0]), case[4]))[0]]
                lb = Labels[q[0][2]]

                if lb == 1:
                    labels = [Labels[c[2]] for c in case[4]]
                    q = list(map(lambda x: (x[0], x[3]), case[4]))
                else:
                    labels = [Labels[q[0][2]]]
                    q = [(q[0][0], q[0][3])]

            for i in range(len(q)):
                if labels[i] == 1:
                    yield (p, q[i][0], (0, 1), q[i][1])  # str, str, int，str
                else:
                    yield (p, q[i][0], (1, 0), q[i][1])
        except:
            pass

# load dictionary


def load_dict(dict_path_word, corpus_path, max_vocab=None, pre_wv=""):
    if os.path.isfile(dict_path_word):
        print("Dict_path=", dict_path_word)
        dict_file_word = open(dict_path_word, "r")
        dict_data_word = dict_file_word.readlines()
        dict_file_word.close()
        if (len(pre_wv) > 0)and(not any(len(x.split()) > 10 for x in dict_data_word)):
            print(dict_data_word[0])
            raise ValueError("No pre-trained wv data.")
    else:
        raise ValueError("No dicts.")
    # word_level
    dict_data_word = list(map(lambda x: x.strip().split(), dict_data_word))  # becomes a list that contains a ID&TOK
    dict_data_word = list(filter(lambda x: len(x) >= 2, dict_data_word))
    dict_data_word = [[i] + x[1:] for i, x in enumerate(dict_data_word)]

    if max_vocab:  # limiting the size of dicitonary
        dict_data_word = dict_data_word[:max_vocab]
    tok2id_word = dict(map(lambda x: (x[1], int(x[0])), dict_data_word))  # token to id
    id2tok_word = dict(map(lambda x: (str(x[0]), x[1]), dict_data_word))  # id to token
    tok2id_word[MARK_UNK] = len(tok2id_word)
    id2tok_word[len(id2tok_word)] = MARK_UNK
    tok2id_word[MARK_PAD] = len(tok2id_word)
    id2tok_word[len(id2tok_word)] = MARK_PAD
    return (tok2id_word, id2tok_word)


def _corpus_map2id(data, tok2id):
    ret = []
    unk = 0
    tot = 0
    for doc in data:
        tmp = []
        for word in doc:
            tot += 1
            try:
                tmp.append(tok2id[word])
            except:
                tmp.append(tok2id["<UNK>"])
                unk += 1
        ret.append(tmp)
    return ret, (tot - unk) / tot

# given a sentence(List[id]), map its ids to tokens


def sen_id2tok(sen, id2tok):
    def change(id2tok, x):
        if str(int(x)) in id2tok:
            return id2tok[str(int(x))]
        else:
            return MARK_UNK
    return "".join(list(map(lambda x: change(id2tok, x), sen)))

# given a sentence(List[wok]), map its tokens to ids


def sen_tok2id(sen, tok2id):
    def change(tok2id, x):
        if x in tok2id:
            return x
        else:
            return MARK_UNK
    sen = list(map(lambda x: change(tok2id, x), sen))
    return list(map(lambda x: tok2id[x], sen))

# p,q,l,respective lengths


def data_feeder(raw_data,  # id_tokenized & shuffled,List[single case] (p,q[i],labels[i])
                batch_size,
                Law,  # tokenized law
                tok2id,
                FLAGS,
                seg=None):

    batch_case = []
    batch_plea = []
    batch_law = []
    l_law = []
    batch_result = []
    # batch_num=len(raw_data)//batch_size
    c = 0
    for case in raw_data:
        if len(case) < 4:
            continue
        c += 1
        if FLAGS.Graph_num == 1:
            batch_case.append(case[0] + case[3] + case[1])
            batch_plea.append(case[1])
            batch_result.append(case[2])
            batch_law.append(case[3])
        elif FLAGS.Graph_num == 2:
            batch_case.append(case[0])
            batch_plea.append(case[1])
            batch_result.append(case[2])
            batch_law.append(case[3])
        else:
            batch_case.append(case[0])
            batch_plea.append(case[1])
            batch_result.append(case[2])
            batch_law.append(case[3])

        if c == batch_size:
            l_case = list(map(lambda x: len(x), batch_case))
            l_plea = list(map(lambda x: len(x), batch_plea))
            l_law = list(map(lambda x: len(x), batch_law))
            for p in range(batch_size):
                while len(batch_case[p]) < max(l_case):
                    batch_case[p].append(tok2id[MARK_PAD])
            for q in range(batch_size):
                while len(batch_plea[q]) < max(l_plea):
                    batch_plea[q].append(tok2id[MARK_PAD])
            for q in range(batch_size):
                while len(batch_law[q]) < max(l_law):
                    batch_law[q].append(tok2id[MARK_PAD])
            yield (np.array(batch_case), np.array(batch_plea), np.array(batch_law), np.array(batch_result), l_case, l_plea, l_law)
            batch_case = []
            batch_plea = []
            batch_result = []
            batch_law = []
            c = 0
            # case plea law result l1 l2 l3


def _data_preparation(raw_data,  # a generator from raw_data_import
                     batch_size,
                     tok2id,
                     all_petition,
                     FLAGS):
    """
    this function read from raw data and produce pre-processed data for training
    """
    if not all_petition:
        default_data_path = "./model/data_id_part{}.dat".format(FLAGS.vocab_size)
    else:
        default_data_path = "./model/data_id_all{}.dat".format(FLAGS.vocab_size)

    if os.path.isfile(default_data_path):
        with open(default_data_path, "rb")as f:
            raw_data = pickle.load(f)
        logging.info("Mapped data loaded.")
    else:
        logging.info("mapping starts")
        raw_data = [x for x in raw_data]
        for i in tqdm.tqdm(range(len(raw_data))):
            raw_data[i] = (sen_tok2id(raw_data[i][0].split(), tok2id),
                           sen_tok2id(raw_data[i][1].split(), tok2id), raw_data[i][2], sen_tok2id(raw_data[i][3].split(), tok2id))
        logging.info("mapping ends")
        with open(default_data_path, "wb") as f:
            pickle.dump(raw_data, f)
    return raw_data  # return mapped data


def get_epoch(raw_data,  # a generator from raw_data_import
              batch_size,
              epoch,
              tok2id,
              all_petition,
              Law,
              FLAGS,
              drop_=0.1,
              length_trial=False,
              random_input=False,
              save_folder_name="",
              restore=""  # using preprocessed data from previous experiment
              ):

    data = _data_preparation(raw_data=raw_data, batch_size=batch_size, tok2id=tok2id, all_petition=all_petition, FLAGS=FLAGS)  # ID-tokenized data

    data_path = "./model/ex_data"
    if not os.path.isdir(data_path):
        os.makedirs(data_path)

    data_file = data_path + "/" + save_folder_name + ".dat"

    if os.path.isfile(data_file) and len(restore) > 0:
        with open(data_file, "rb")as f:
            data, test_set = pickle.load(f)
            logging.info("data restored from %s" % save_folder_name)
            yield len(data)
            yield test_set

    else:

        y = -int(len(data) * drop_)

        data.sort(key=lambda x: len(x[0]))
        data = data[:y]
        data = data[12000:]
        logging.info("Max docement length: %d." % len(data[-1][0]))
        logging.info("%d documents in total." % len(data))

        shuffle(data)
        test_set = data[int(len(data) * 0.9):]
        valid_set = data[int(len(data) * 0.8):int(len(data) * 0.9)]
        if not length_trial and not random_input:
            test_set.sort(key=lambda x: len(x[0]) + randint(0, 10))
        elif random_input:
            shuffle(test_set)
        data = data[:int(len(data) * 0.8)]
        test_set = (test_set, batch_size, tok2id)
        yield len(data)
        yield test_set
        with open(data_file, "wb")as f:
            pickle.dump((data, test_set), f)

    for i in range(epoch):
        if random_input:
            shuffle(data)
        else:
            data.sort(key=lambda x: len(x[0]) + randint(0, 30 * (1 - int(length_trial)) + int(length_trial)), reverse=length_trial)
        yield data_feeder(raw_data=data, batch_size=batch_size, tok2id=tok2id, Law=Law, FLAGS=FLAGS)
