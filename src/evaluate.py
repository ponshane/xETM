import json
import math
import re
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
from gensim import corpora
from gensim.matutils import corpus2csc
from scipy.spatial import distance
from scipy.stats import entropy
from sklearn.feature_extraction.text import CountVectorizer

from .preprocessing.format_txt_into_mat import _create_matrixes, split_bow

"""
The utils code for compute c-npmi
"""


## check language
def check_language(word):
    # 以正規表示法判斷語言
    try:
        # 可以解碼為 ascii 的為英文單字
        word.encode(encoding="utf-8").decode("ascii")
        return "en"
    except UnicodeDecodeError:
        # 為什麼還有一正規判斷?
        if re.search("[\u4e00-\u9fff]", word):
            return "zh"
        else:
            return None


def split_topk_topic_word(topic_word_matrix, topk=10):
    en_topic = []
    cn_topic = []

    for words in topic_word_matrix:
        words_arr = np.array(words)
        lang_labels = np.array(list(map(check_language, words)))
        feature_en_idx = np.where(lang_labels == "en")[0]
        feature_zh_idx = np.where(lang_labels == "zh")[0]
        # en topk word
        en_topk_words = words_arr[feature_en_idx][:topk]
        en_topic.append(en_topk_words.tolist())
        # cn topk word
        cn_topk_words = words_arr[feature_zh_idx][:topk]
        cn_topic.append(cn_topk_words.tolist())
    return en_topic, cn_topic


## NPMI
### code from 家暄學長<https://github.com/ponshane/CLTM/blob/master/src/codebase/topic_evaluator.py>
def documents_to_cooccurence_matrix(
    source_language_documents, target_language_documents
):
    """
    Reference: code from 家暄學長<https://github.com/ponshane/CLTM/blob/master/src/codebase/topic_evaluator.py>
    """

    compound_documents = [
        doc_in_source + doc_in_target
        for doc_in_source, doc_in_target in zip(
            source_language_documents, target_language_documents
        )
    ]

    # turn into gensim's corpora
    compound_dictionary = corpora.Dictionary(compound_documents)
    compund_corpus = [compound_dictionary.doc2bow(text) for text in compound_documents]

    # transform into term_document matrix, each element represents as frequency
    term_document_matrix = corpus2csc(compund_corpus)

    # 利用 corpus2csc 轉換後每個元素為該詞於該篇的詞頻(會大於1)，但 umass score 需要的是 the count of documents containing the word
    # 因此得利用 np.where 重新轉換矩陣，使每個元素單純標記該詞是否出現於該篇(1 or 0)
    # np.where 無法在 csc matrix 故使用以下解決
    term_document_matrix[term_document_matrix >= 1] = 1
    cooccurence_matrix = term_document_matrix @ term_document_matrix.T

    print(f"type of cooccurence_matrix: {type(cooccurence_matrix)}")
    print(
        f"shape of tdm and cooccurence_matrix: {(term_document_matrix.shape, cooccurence_matrix.shape)}"
    )
    return (
        cooccurence_matrix,
        term_document_matrix,
        compound_dictionary,
        len(compund_corpus),
    )


def NPMI(cooccurence_matrix, word_i, word_j, num_of_documents):
    epsilon = 1e-12
    co_count = cooccurence_matrix[word_i, word_j] / num_of_documents
    single_count_i = cooccurence_matrix[word_i, word_i] / num_of_documents
    single_count_j = cooccurence_matrix[word_j, word_j] / num_of_documents
    pmi = math.log((co_count + epsilon) / (single_count_i * single_count_j))
    return pmi / (math.log(co_count + epsilon) * (-1))


def npmi_score(
    cn_topic,
    en_topic,
    topk,
    cooccurence_matrix,
    compound_dictionary,
    num_of_documents,
    coherence_method,
):
    """
    Input: list of list: cn_topic(中文分群結果), en_topic(英文分群結果); scalar: topk(衡量topk個字); matrix: cooccurence_matrix
    (儲存每個字出現次數和兩兩個字的共同出現次數 － 以篇為單位); compound_dictionary: gensim 的字典
    Output: umass, npmi coherence score
    Reference:
        1) http://qpleple.com/topic-coherence-to-evaluate-topic-models/
        2) Mimno, D., Wallach, H. M., Talley, E., Leenders, M., & McCallum, A. (2011, July). Optimizing semantic coherence in topic models.
           #In Proceedings of the conference on empirical methods in natural language processing (pp. 262-272). Association for Computational Linguistics.
    Issue:
        1) [Solve!] Original metric uses count of documents containing the words
    """
    each_topic_coher = []
    for ctopic, etopic in zip(cn_topic, en_topic):
        # below two assertion is very important because
        # 1) minor problem split_language method is a risky method because it may strips some words
        # 2) continue LDAs can not promise to produce the same vocabularies size across languages,
        #    and be a extreme imbalance distribution. (單語言主題群，僅有少數跨語言詞彙)

        assert len(ctopic) >= topk
        assert len(etopic) >= topk

        cn_idx = [
            compound_dictionary.token2id[cn]
            for cn in ctopic[:topk]
            if cn in compound_dictionary.token2id
        ]
        en_idx = [
            compound_dictionary.token2id[en]
            for en in etopic[:topk]
            if en in compound_dictionary.token2id
        ]

        """
        debug line
        print(ctopic[:topk])
        print(etopic[:topk])
        """

        coherences = []
        for each_cn in cn_idx:
            for each_en in en_idx:
                if coherence_method == "umass":
                    # calculate_umass_score_between_two_words
                    co_count = cooccurence_matrix[each_cn, each_en]
                    single_count = cooccurence_matrix[each_en, each_en]
                    pmi = math.log((co_count + 1) / single_count)
                    coherences.append(pmi)
                elif coherence_method == "npmi":
                    npmi = NPMI(cooccurence_matrix, each_cn, each_en, num_of_documents)
                    coherences.append(npmi)

        each_topic_coher.append(sum(coherences) / len(coherences))

    return sum(each_topic_coher) / len(each_topic_coher)


"""
diversity util
"""


def get_topic_diversity(beta, topk):
    num_topics = beta.shape[0]
    list_w = np.zeros((num_topics, topk))
    for k in range(num_topics):
        idx = beta[k, :].cpu().numpy().argsort()[-topk:][::-1]
        list_w[k, :] = idx
    n_unique = len(np.unique(list_w))
    TD = n_unique / (topk * num_topics)
    print("Topic diveristy is: {}".format(TD))


class Evaluator:
    def __init__(self, args, vocab, model, words_all) -> None:
        self.args = args
        self.vocab = vocab
        self.model = model
        self.words_all = words_all
        self.evaluate_value = dict()

    def inference_all(self):
        self.infer_theta_full_corpus()
        self.get_purity_df()
        self.compute_en_purity()
        self.compute_zh_purity()
        self.get_reflexibility()
        self.get_cnpmi()
        self.get_topic_diversity()
        save_dir = Path(self.args.load_from)
        with (save_dir.parents[0] / "evaluate_value.json").open("w") as f:
            json.dump(self.evaluate_value, f)
        pprint(self.evaluate_value)
        print("Finish Theta Inference.")

    def infer_theta_full_corpus(self):
        ## load full docs
        with open(self.args.full_data_path, "r") as f:
            docs = [line.strip().split() for line in f.readlines()]
        self.docs = docs

        docs_bow = [
            list(filter(lambda x: x is not None, map(self.vocab.get, doc)))
            for doc in docs
        ]

        bow_dict = _create_matrixes(self.vocab, {"full": docs_bow})
        bow = bow_dict["full"]
        bow_tokens, bow_counts = split_bow(bow, bow.shape[0])

        token_arr = [np.array(doc).reshape(1, len(doc)) for doc in bow_tokens]
        tmp_arr = np.empty(len(token_arr), object)
        tmp_arr[:] = token_arr
        full_tokens = tmp_arr

        count_arr = [np.array(doc).reshape(1, len(doc)) for doc in bow_counts]
        tmp_arr = np.empty(len(count_arr), object)
        tmp_arr[:] = count_arr
        full_counts = tmp_arr

        ## infer full corpus
        thetas = self.model.infer_theta(self.args, full_tokens, full_counts)
        self.thetas = thetas.cpu().numpy()
        # print(thetas[0], thetas[0].shape)
        print(thetas.shape)

        # save thetas
        np.save(
            f"./experiment/thetas_K_{self.args.num_topics}_{self.args.suffix}_full",
            self.thetas,
        )

    def get_purity_df(self):
        # compute purity
        with open(self.args.data_label_path) as f:
            college_label = [label.strip() for label in f.readlines()]

        df = pd.DataFrame(
            {
                "topic_labels": self.thetas.argmax(axis=1),
                "college_labels": college_label,
            }
        )
        purity_results = (
            self.compute_purity(df)
            .set_index("topic")
            .join(df.topic_labels.value_counts().to_frame())
            .sort_values("topic_labels", ascending=False)
        )
        print(purity_results)
        # df_purity = compute_purity(df).set_index("topic")
        # all
        margin_topic_distri = np.nansum(self.thetas, axis=0) / np.nansum(self.thetas)
        purity_results["margin"] = margin_topic_distri
        topic_distri = purity_results.groupby("topic_label")["margin"].sum()
        self.df = df
        self.purity_results = purity_results
        self.evaluate_value["purity"] = purity_results["purity"].mean()
        self.topic_distri = topic_distri

    def compute_en_purity(self):
        # en purity
        half = int(self.df.shape[0] / 2)
        en_df_purity = self.compute_purity(self.df.iloc[:half, :]).set_index("topic")
        en_thetas = self.thetas[:half, :]
        en_margin_topic_distri = np.nansum(en_thetas, axis=0) / np.nansum(en_thetas)
        en_margin_topic_distri = pd.DataFrame({"en_margin": en_margin_topic_distri})
        en_margin_topic_distri.index.name = "topic"
        en_df_purity = en_df_purity.join(en_margin_topic_distri)
        en_topic_distri = en_df_purity.groupby("topic_label")["en_margin"].sum()
        en_topic_distri = en_topic_distri / en_topic_distri.sum()
        self.en_df_purity = en_df_purity
        self.en_topic_distri = en_topic_distri

    def compute_zh_purity(self):
        # zh purity
        half = int(self.df.shape[0] / 2)
        zh_df_purity = self.compute_purity(self.df.iloc[half:, :]).set_index("topic")
        # zh
        zh_thetas = self.thetas[half:, :]
        zh_margin_topic_distri = np.nansum(zh_thetas, axis=0) / np.nansum(zh_thetas)
        zh_margin_topic_distri = pd.DataFrame({"zh_margin": zh_margin_topic_distri})
        zh_margin_topic_distri.index.name = "topic"
        zh_df_purity = zh_df_purity.join(zh_margin_topic_distri)
        zh_topic_distri = zh_df_purity.groupby("topic_label")["zh_margin"].sum()
        zh_topic_distri = zh_topic_distri / zh_topic_distri.sum()
        self.zh_df_purity = zh_df_purity
        self.zh_topic_distri = zh_topic_distri

    def compute_purity(self, df_purity):
        """df columns 包括 topic_labels & college_labels"""
        pg = df_purity.groupby("topic_labels")
        group_distribution = []
        result_purity = []
        for group, df in pg:
            df1 = (
                df["college_labels"]
                .value_counts()
                .to_frame()
                .rename(columns={"college_labels": "frequency"})
            )
            df_proportion = df1.join(
                df["college_labels"]
                .value_counts(normalize=True)
                .to_frame()
                .rename(columns={"college_labels": "proportion"})
            )
            group_distribution.append(
                f"topic: {group} {df1.idxmax().values[0]}\n{df_proportion.to_string()}\n"
            )
            result_purity.append(
                [
                    group,
                    df1.idxmax().values[0],
                    (df1.max().values[0] / df.shape[0]),
                ]
            )

        result_purity = pd.DataFrame(
            result_purity, columns=["topic", "topic_label", "purity"]
        )
        return result_purity

    def get_reflexibility(self):
        # JSD of topic distribution
        # output the topic predict distribution
        true_distribution = pd.DataFrame(
            {"true_frequency": [31302, 7227, 93623, 16722, 6011, 8277]},
            index=["商管學院", "文學院", "理工電資學院", "生物資源暨農學院", "社會科學院", "醫學院"],
        )
        true_distribution["true_proportion"] = true_distribution.true_frequency / sum(
            true_distribution.true_frequency
        )
        true_distribution = true_distribution.join(self.topic_distri).fillna(0)
        true_distribution = true_distribution.join(self.en_topic_distri).fillna(0)
        true_distribution = true_distribution.join(self.zh_topic_distri).fillna(0)

        print("true proportion:")
        print(true_distribution["margin"].to_frame().T)

        reflexibility = distance.jensenshannon(
            true_distribution["true_proportion"], true_distribution["margin"]
        )
        print(f"the reflexibility (JSD) of model to true distribution: {reflexibility}")

        print(true_distribution["true_proportion"].to_frame().T)
        print("-" * 100)

        en_reflexibility = distance.jensenshannon(
            true_distribution["true_proportion"], true_distribution["en_margin"]
        )
        print(
            f"the reflexibility (JSD) of model to en true distribution: {en_reflexibility}"
        )
        print(true_distribution["en_margin"].to_frame().T)
        print("-" * 100)
        zh_reflexibility = distance.jensenshannon(
            true_distribution["true_proportion"], true_distribution["zh_margin"]
        )
        print(
            f"the reflexibility (JSD) of model to zh true distribution: {zh_reflexibility}"
        )
        print(true_distribution["zh_margin"].to_frame().T)
        self.evaluate_value["reflexibility"] = reflexibility

    def get_cnpmi(self, topk=10):
        en_topic, cn_topic = split_topk_topic_word(
            topic_word_matrix=self.words_all, topk=topk
        )
        half_docs = int(len(self.docs) / 2)
        self.docs_en = self.docs[:half_docs]
        self.docs_ch = self.docs[half_docs:]
        (
            cooccurence_matrix,
            _,
            compound_dictionary,
            num_of_documents,
        ) = documents_to_cooccurence_matrix(
            source_language_documents=self.docs_en,
            target_language_documents=self.docs_ch,
        )
        self.evaluate_value["c-npmi"] = npmi_score(
            cn_topic=cn_topic,
            en_topic=en_topic,
            topk=topk,
            cooccurence_matrix=cooccurence_matrix,
            compound_dictionary=compound_dictionary,
            num_of_documents=num_of_documents,
            coherence_method="npmi",
        )
        print(f"c-npmi score: {self.evaluate_value.get('c-npmi')}")

    def get_topic_diversity(self):
        diversity = []
        for topk in [10, 50, 100, 200, 500]:
            value = len(np.unique([word[:topk] for word in self.words_all])) / (
                topk * 50
            )
            print(f"diversity {topk}: {value}")
            diversity.append(value)
        self.evaluate_value["diversity"] = diversity

    def entropy_JSD_of_theta_pair(self):
        print(f"entropy of theta:")
        print(pd.Series(entropy(self.thetas, axis=1)).describe())
        print(f"average topic share:")
        print(np.nanmean(self.thetas, axis=0))

        jsd = distance.jensenshannon(
            self.thetas[np.arange(0, int(self.thetas.shape[0] / 2)), :],
            self.thetas[
                np.arange(int(self.thetas.shape[0] / 2), self.thetas.shape[0]), :
            ],
            axis=1,
        )
        print("JSD:")
        print(pd.Series(jsd).describe())
        print("the accuary of max topic of comparable article: ")
        print(
            np.mean(
                self.thetas[np.arange(0, int(self.thetas.shape[0] / 2)), :].argmax(
                    axis=1
                )
                == self.thetas[
                    np.arange(int(self.thetas.shape[0] / 2), self.thetas.shape[0]), :
                ].argmax(axis=1)
            )
        )
        print(f"na article: {np.isnan(self.thetas.sum(axis=1)).sum()}")

    def the_max_aggrement(self):
        ## the max agreement of each docs
        rows = np.arange(self.thetas.shape[0])
        print("the distribution max proportion topic of each docs")
        print(
            pd.DataFrame(self.thetas[rows, self.thetas.argmax(axis=1)])
            .describe()
            .to_string()
        )
