"""
Target:
1. generate word_vector
1.5 tf-idf
2. loading case documents and law text
3. calculating distance
4. extract most relevant law articles
"""

import numpy as np
import os
import pickle
import re
import tqdm
from gensim.models import Word2Vec
import jieba.analyse
import jieba


MARK_UNK = "<UNK>"
MARKS = [MARK_UNK]


def WordVector(W2V_Model_Path,
               Corpus_path,
               EmbeddingSize=100):
    """
    This function either train/load word vectors with a corpus
    :param W2V_Model_Path:   the path of the file for pretraiend word vectors
    :param Corpus_path:      the path of the corpus used to train word vectors, the corpus should be processed with word-segmentation in advance.
    :param EmbeddingSize
    :return: model
    """

    # load or train

    assert W2V_Model_Path

    if os.path.isfile(W2V_Model_Path):
        # load from pre-existing models
        model = Word2Vec.load(W2V_Model_Path)  # model.wv[token]   ndarray(1*embed)
        print("Model loaded.")
    else:
        assert Corpus_path
        with open(Corpus_path, 'r')as f:
            sentences = f.readlines()
            assert (sentences[0].count(' ') + sentences[10].count(' ') + sentences[100].count(' ')) /\
                len(sentences[0] + sentences[10] + sentences[100]) > 0.1  # a simple heuristics to check if the documents have been segmented
            sentences = [sentence.strip().split() for sentence in sentences]
        model = Word2Vec(sentences, EmbeddingSize, workers=10)
        model.save(W2V_Model_Path)
        print("Model trained, and saved to %s." % W2V_Model_Path)

    return model


def TFIDF(Corpus_path):
    """
    :param Corpus_path: do not need to be segmented
    :return:
    """
    KeyWord = jieba.analyse.extract_tags(open(Corpus_path, 'r').read(),
                                         topK=20000, withWeight=True)
    return {k: v for k, v in KeyWord}


def extraction(W2V_Model_Path,
               Corpus_path,
               EmbeddingSize=100,
               CaseFileDirectory="./LRCraw",
               filenames="law_seg_refined_data_new_0_30000/law_seg_refined_data_new_30000_60000/law_seg_refined_data_new_60000_90000/law_seg_refined_data_new_90000_120000".split("/"),
               law_text="./LRCraw/divorce_full.txt"):
    """
    :param CaseFileDirectory:
    :param filenames:
    :param law_text: law articles are not required to be segmented in advance
    :return:
    """
    print("Start to load law text.")
    with open(law_text, "r")as f:
        laws = f.readlines()
        laws = list(map(lambda x: list(filter(lambda x: len(x) > 0, list(jieba.cut(x.strip().replace(' ', ''), cut_all=False)))), laws))
    print("law articles loaded.")

    div = laws[:26]
    child = laws[26:39]
    money = laws[39:]

    # 读入WV, tf-idf weight
    print("Start to load dictionary.")
    WVmodel = WordVector(W2V_Model_Path=W2V_Model_Path,
                         Corpus_path=Corpus_path,
                         EmbeddingSize=EmbeddingSize)
    weight = TFIDF(Corpus_path=Corpus_path)
    print("finished")

    for file in filenames:
        with open(os.path.join(CaseFileDirectory, file), "rb")as f:
            raw_data = pickle.load(f)
        for idx in tqdm.tqdm(range(len(raw_data))):
            for q in range(len(raw_data[idx][4])):
                raw_data[idx][4][q][-1].clear()
                raw_data[idx][4][q][-1].extend(law_extract(raw_data[idx][0][0],
                                                           raw_data[idx][4][q][0],
                                                           [div, child, money],
                                                           WVmodel,
                                                           weight))
        with open(os.path.join(CaseFileDirectory, "law_" + file), "wb")as f:
            pickle.dump(raw_data, f)


def dis(a, b, WV, weight):
    w = a.strip().split()
    w_vec = []
    for word in w:
        try:
            x = WV[word] * 10000 / min(weight[word], 1e5)
            w_vec.append(x)
        except:
            pass

    if not len(w_vec):
        return 0

    w_vec = sum(w_vec)

    g = b.strip().split()
    g_vec = []
    for word in g:
        try:
            x = WV[word] * 10000 / min(weight[word], 1e5)
            g_vec.append(x)
        except:
            pass

    if not len(g_vec):
        return 0

    g_vec = sum(g_vec)

    return np.dot(w_vec, g_vec) / (np.sqrt(np.dot(w_vec, w_vec)) * np.sqrt(np.dot(g_vec, g_vec)))


def law_extract(fact, query, law_text, WV, weight):
    """
    fact: str,
    law_text: List[str]
    """
    s1 = ["(^[^，。]*?离婚[^，。]*?$)|(^[^，。]*?解除[^，。]*?婚姻[^，。]*?$)", "((不准)|(驳回))[^，。]*?离婚", "驳回"]
    s2 = ["(^[^，。]*?离婚[^，。]*?$)|([^，。]*?解除[^，。]*?婚姻[^，。]*?$)", "^准{0,1}[^，。]*?离婚", "准许"]
    s_div = [s1, s2]
    # 子女归属：
    s1 = ["((孩子{0,1})|(小孩))[^。]*?[由随归][^。]*?((我)|(原告))[^，。]*?((抚养$)|(抚育$)|(抚养[^费])|(抚育[^费])|(生活))",
          "((孩子{0,1})|(小孩))[^。]*?[由随归][^。]*?((我)|(原告))[^，。]*?((抚养$)|(抚育$)|(抚养[^费])|(抚育[^费])|(生活))", "准许"]
    s2 = ["((儿子)|(\S子))[^。]*?[由随归][^。]*?((我)|(原告))[^，。]*?((抚养$)|(抚育$)|(抚养[^费])|(抚育[^费])|(生活))",
          "((儿子)|(\S子)|(孩子{0,1})|(小孩))[^。]*?[由随归][^。]*?((我)|(原告))[^，。]*?((抚养$)|(抚育$)|(抚养[^费])|(抚育[^费])|(生活))", "准许"]
    s3 = ["((\S{0,1}女儿{0,1}))[^。]*?[由随归][^。]*?((我)|(原告))[^，。]*?((抚养$)|(抚育$)|(抚养[^费])|(抚育[^费])|(生活))",
          "((\S{0,1}女儿{0,1})|(孩子{0,1})|(小孩))[^。]*?[由随归][^。]*?((我)|(原告))[^，。]*?((抚养$)|(抚育$)|(抚养[^费])|(抚育[^费])|(生活))",
          "准许"]
    s4 = ["((孩子{0,1})|(小孩))[^。]*?[由随归][^。]*?被告[^，。告]*?((抚养$)|(抚育$)|(抚养[^费])|(抚育[^费])|(生活))",
          "((孩子{0,1})|(小孩))[^。]*?[由随归][^。]*?被告[^，。告]*?((抚养$)|(抚育$)|(抚养[^费])|(抚育[^费])|(生活))", "准许"]
    s5 = ["((儿子)|(\S子))[^。]*?[由随归][^。]*?被告[^，。告]*?((抚养$)|(抚育$)|(抚养[^费])|(抚育[^费])|(生活))",
          "((儿子)|(\S子)|(孩子{0,1})|(小孩))[^。]*?[由随归][^。]*?被告[^，。告]*?((抚养$)|(抚育$)|(抚养[^费])|(抚育[^费])|(生活))", "准许"]
    s6 = ["((\S{0,1}女儿{0,1}))[^。]*?[由随归][^。]*?被告[^，。告]*?((抚养$)|(抚育$)|(抚养[^费])|(抚育[^费])|(生活))",
          "((\S{0,1}女儿{0,1})|(孩子{0,1})|(小孩))[^。]*?[由随归][^。]*?被告[^，。告]*?((抚养$)|(抚育$)|(抚养[^费])|(抚育[^费])|(生活))", "准许"]
    s7 = ["((原告)|(我))抚养[^，。告]*?子",
          "((\S{0,1}女儿{0,1})|(儿子)|(\S子)|(孩子{0,1})|(小孩))[^。]*?[由随归][^。]*?((我)|(原告))[^，。]*?((抚养$)|(抚育$)|(抚养[^费])|(抚育[^费])|(生活))",
          "准许"]
    s8 = ["((被告)|(我))抚养[^，。告]*?子",
          "((\S{0,1}女儿{0,1})|(儿子)|(\S子)|(孩子{0,1})|(小孩))[^。]*?[由随归][^。]*?((我)|(被告))[^，。]*?((抚养$)|(抚育$)|(抚养[^费])|(抚育[^费])|(生活))",
          "准许"]
    s_ch = [s1, s2, s3, s4, s5, s6, s7, s8]
    # 抚养费：
    s1 = ["(((抚养费)|(抚育费))[^。]*?((各自)|(分担)))|(((各自)|(分担))[^。]*?((抚养费)|(抚育费)))",
          "(((抚养费)|(抚育费))[^。]*?((各自)|(分担)))|(((各自)|(分担))[^。]*?((抚养费)|(抚育费)))", "准许"]
    s2 = [
        "((原告)|(我))[^。告]*?(向\S告){0,1}[^。告]*?((支付)|(给付)|(负担)|(承担)|(负责)|(给予))[^，。告]*?((生活费)|(抚养费)|(补助)|(抚育费))[^。无不]*?([0-9.]*元*){0,1}",
        "((原告)|(我))[^。无不告]*?(向\S告){0,1}[^。告无不]*?((支付)|(给付)|(负担)|(承担)|(负责)|(给予))[^，。无不]*?((生活费)|(抚养费)|(补助)|(抚育费))[^。无不告]*?([0-9.]*元*){0,1}",
        "准许"]
    s3 = ["由[^。无不]*?((原告)|(我))((支付)|(给付)|(负担)|(承担)|(负责)|(给予))[^。无不]*?((生活费)|(抚养费)|(补助)|(抚育费))[^，。无不]*?([0-9.]*元*){0,1}",
          "由[^。无不]*?((原告)|(我))((支付)|(给付)|(负担)|(承担)|(负责)|(给予))[^。无不]*?((生活费)|(抚养费)|(补助)|(抚育费))[^，。无不]*?([0-9.]*元*){0,1}",
          "准许"]
    s4 = ["((生活费)|(抚养费)|(补助)|(抚育费))[^。告]*?由[^。告]*?((原告)|(我))[^。告]*?((支付)|(给付)|(负担)|(承担)|(负责)|(给予))",
          "((生活费)|(抚养费)|(补助)|(抚育费))[^。无不告]*?由[^。无不告]*?((原告)|(我))[^。无不告]*?((支付)|(给付)|(负担)|(承担)|(负责)|(给予))", "准许"]
    s5 = [
        "被告[^。告]*?(向\S告){0,1}[^。告]*?((支付)|(给付)|(负担)|(承担)|(负责)|(给予))[^，。告]*?((生活费)|(抚养费)|(补助)|(抚育费))[^。]*?([0-9.]*元*){0,1}",
        "被告[^。无不告]*?(向\S告){0,1}[^。告无不]*?((支付)|(给付)|(负担)|(承担)|(负责)|(给予))[^，。无不]*?((生活费)|(抚养费)|(补助)|(抚育费))[^。无告不]*?([0-9.]*元*){0,1}",
        "准许"]
    s6 = ["由[^。告]*?被告((支付)|(给付)|(负担)|(承担)|(负责)|(给予))[^。告]*?((生活费)|(抚养费)|(补助)|(抚育费))[^，。]*?([0-9.]*元*){0,1}",
          "由[^。无告不]*?被告((支付)|(给付)|(负担)|(承担)|(负责)|(给予))[^。无告不]*?((生活费)|(抚养费)|(补助)|(抚育费))[^，。无不]*?([0-9.]*元*){0,1}", "准许"]
    s7 = ["((生活费)|(抚养费)|(补助)|(抚育费))[^。告]*?由[^告。]*?被告[^。告]*?((支付)|(给付)|(负担)|(承担)|(负责)|(给予))",
          "((生活费)|(抚养费)|(补助)|(抚育费))[^。无不告]*?由[^。无告不]*?被告[^。无不告]*?((支付)|(给付)|(负担)|(承担)|(负责)|(给予))", "准许"]
    s8 = ["((抚养费)|(抚育费))自理", "((抚养费)|(抚育费))自理", "准许"]
    s_transfer = [s1, s2, s3, s4, s5, s6, s7, s8]

    law = ""

    for s in s_div:
        if re.search(s[0], query):
            law = law_text[0]
            break
    if law == "":
        for s in s_ch:
            if re.search(s[0], query):
                law = law_text[1]
                break
    if law == "":
        for s in s_transfer:
            if re.search(s[0], query):
                law = law_text[2]
                break
    if law == "":
        return " 。 ".join(law_text[0][:4] + law_text[1][:4] + law_text[2][:4])

    a = law[:4]
    law = law[4:]
    law.sort(key=lambda x: dis(x, fact + query, WV, weight), reverse=True)
    return " 。 ".join(a + law[:4])


if __name__ == "__main__":
    pass
    #extraction(W2V_Model_Path, Corpus_path)
