import numpy as np
import os
import sys
import pickle
import re
import tqdm
import logging
import subprocess
from collections import Counter
from random import shuffle, randint
from multiprocessing import Pool
from . import LawArticleExtraction


MARK_UNK = "<UNK>"
MARKS = [MARK_UNK]


def build_corpus(raw_path="./raw",
                 filenames="data_new_0_30000/data_new_30000_60000/data_new_60000_90000/data_new_90000_120000".split(
                     "/"),
                 law_text="./raw/divorce_full.txt",
                 corpus="./model/Corpus.txt",
                 length=0):
    """
    生成语料库
    """
    # 生成语料库
    print("Starts to load data.")
    raw_data = []
    for file in filenames:
        with open(os.path.join(raw_path, file), "rb") as f:
            raw_data += pickle.load(f)
    print("Cases loaded.")
    with open(law_text, "r") as f:
        law = f.read().replace("\n", "")
    print("Law text loaded.")
    Corpus = law + "。".join(
        [case[0][0] + "。".join(["。".join([i for i in c if i]) for c in case[4]]) for case in raw_data])
    Corpus = Corpus.replace("\n", "").replace(" ", "")
    with open(corpus, "w") as f:
        f.write(Corpus[length:])
    print("Corpus built.")


def segmentation(corpus="./model/Corpus.txt",
                 other_corpus="./raw/zhwiki.txt",
                 seg_corpus="./model/seg_Corpus.txt",
                 path="./model/",
                 bulk=8):
    assert (os.path.isfile(corpus))

    with open(corpus, "r") as f:
        Corpus = f.read().replace("\n", "")
    print("Law corpus loaded.")
    if other_corpus == "default":
        other_corpus = "./raw/zhwiki.txt"
    if len(other_corpus) > 5:
        with open(other_corpus, "r") as f:
            other = f.read().replace("\n", "")
        full_corpus = Corpus + Corpus + other + Corpus
    else:
        other = ""
        full_corpus = Corpus

    print("Corpus pre-processed. Start to do Segmentation.")

    # slicing
    length = len(full_corpus) // bulk
    for i in range(bulk):
        with open(path + "Corpus_part_" + str(i) + ".txt", "w")as f:
            f.write(full_corpus[length * i:length * (i + 1)].strip())
    print("Corpus has been sliced into %d piecies." % bulk)

    p = Pool(bulk)
    for i in range(bulk):
        p.apply_async(subprocess.call, args=(["python3", "seg.py", path + "Corpus_part_" + str(i) + ".txt"],))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')

    print("Corpus pieces segmented.")

    seg_corpuses = ""

    for i in range(bulk):
        with open(path + "Corpus_part_" + str(i) + ".txt", "r")as f:
            seg_corpuses += " " + f.read().strip()
        subprocess.call(["rm", path + "Corpus_part_" + str(i) + ".txt"], )

    print("Pieces aggregated.")

    with open(seg_corpus, "w") as f:
        f.write(seg_corpuses)

    print("Corpuses segmentation finished.")


def create_dict(corpus_path="./model/seg_Corpus.txt",
                dict_path="./model",
                dict_name="Dict.txt",
                max_vocab=None,
                bulk=8):
    print("Start to build a new dictionary.")

    with open(corpus_path, "r") as f:
        corpus_ = f.read()
    print("Corpus loaded.")

    length = len(corpus_) // bulk
    for i in range(bulk):
        with open(dict_path + "/Dict_part_" + str(i) + ".txt", "w")as f:
            f.write(corpus_[length * i:length * (i + 1)].strip())
    print("Corpus has been sliced into %d piecies." % bulk)

    p = Pool(bulk)
    for i in range(bulk):
        p.apply_async(subprocess.call, args=(["python3", "count.py", dict_path + "/Dict_part_" + str(i) + ".txt"],))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')

    words = dict()

    for i in range(bulk):
        with open(dict_path + "/Dict_part_" + str(i) + ".txt", "r")as f:
            content = f.readlines()
            content = list(map(lambda x: x.strip().split(), content))
            content = list(map(lambda x: (x[0], int(x[1])), content))
        for pair in content:
            words[pair[0]] = words.get(pair[0], 0) + pair[1]
        subprocess.call(["rm", dict_path + "/Dict_part_" + str(i) + ".txt"], )

    words = [(x, y) for x, y in words.items()]
    words.sort(key=lambda x: x[1], reverse=True)
    words = [word for word in words if not re.search("^[0-9]+.{0,1}[0-9]*$", word[0])]

    if max_vocab:
        words = words[:max_vocab]

    for mark in MARKS:
        words.append(mark)

    print("Dict built.")

    with open(os.path.join(dict_path, dict_name), 'w') as dict_file:
        for idx, tok in enumerate(words):
            print(idx, tok[0], tok[1], file=dict_file)  # id, token, for each line

    print("Dict saved.")


def AddLawArticles():
    pass


def Mapping():
    pass


if __name__ == "__main__":
    pass
