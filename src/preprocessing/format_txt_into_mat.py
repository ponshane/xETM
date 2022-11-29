from .operations import contains_numeric, contains_punctuation, remove_empty,\
    create_list_word_indices, create_doc_indices, create_bow, split_bow
from sklearn.feature_extraction.text import CountVectorizer
from scipy.io import savemat
import numpy as np
import os
import pickle

def _normalize_txt(init_docs):
    # remove punctuation & lowerize characters
    init_docs = [[w.lower() for w in init_docs[doc] if not contains_punctuation(w)] for doc in range(len(init_docs))]
    # remove numeric
    init_docs = [[w for w in init_docs[doc] if not contains_numeric(w)] for doc in range(len(init_docs))]
    # remove single character, e.g., "a", "b"
    init_docs = [[w for w in init_docs[doc] if len(w)>1] for doc in range(len(init_docs))]
    # unnest the nested list into format ["tokens of first document 1". "......"]
    init_docs = [" ".join(init_docs[doc]) for doc in range(len(init_docs))]

    return init_docs

def _initialize_vocabulary_dict(init_docs, max_df, min_df):
    print('counting document frequency of words...')
    cvectorizer = CountVectorizer(min_df=min_df, max_df=max_df, stop_words=None)
    # cvz is a document-term matrix
    cvz = cvectorizer.fit_transform(init_docs).sign()

    print('building the vocabulary...')
    sum_counts = cvz.sum(axis=0) # calculate (axis=0) frequency of each term
    v_size = sum_counts.shape[1] # size of vocabulary
    sum_counts_np = np.zeros(v_size, dtype=int)
    for v in range(v_size):
        sum_counts_np[v] = sum_counts[0,v]
    # word2id, a dicitonary maps word to id
    word2id = dict([(w, cvectorizer.vocabulary_.get(w)) for w in cvectorizer.vocabulary_])
    # id2word, a dicitonary maps id to word
    id2word = dict([(cvectorizer.vocabulary_.get(w), w) for w in cvectorizer.vocabulary_])
    del cvectorizer
    print('\tinitial vocabulary size: {}'.format(v_size))

    # sort words in vocabulary, which put the more frequent words into a more top positions 
    idx_sort = np.argsort(sum_counts_np) # return index of original vocab that would be used for sorting the dictionary later
    vocab_aux = [id2word[idx_sort[cc]] for cc in range(v_size)] # return a list of sorted terms

    # create dictionary and inverse dictionary
    vocab = vocab_aux
    del vocab_aux
    word2id = dict([(w, j) for j, w in enumerate(vocab)])
    # id2word = dict([(j, w) for j, w in enumerate(vocab)])

    return word2id

def _split_data(init_docs, word2id, ratio):

    num_docs = len(init_docs)

    # Split in train/test/valid
    print('tokenizing documents and splitting into train/test/valid...')
    trSize = int(np.floor(ratio["training"]*num_docs)) # size of training
    tsSize = int(np.floor(ratio["testing"]*num_docs)) # size of testing
    vaSize = int(num_docs - trSize - tsSize) # size of validation
    assert num_docs == trSize + tsSize + vaSize

    # np.random.permutation randomly generates a list of indexes (i.e., idx_permute)
    # idx_permute is then used for split training, validation , and testing dataset
    idx_permute = np.random.permutation(num_docs).astype(int)
    """ example
    >>> np.random.permutation(10)
    array([1, 7, 4, 3, 0, 9, 2, 5, 8, 6]) # random
    """

    # remove words not in train_data
    # as you can see, idx_permute[idx_d] is used for picking the document in init_docs
    vocab = list(set([w for idx_d in range(trSize) for w in init_docs[idx_permute[idx_d]].split() if w in word2id]))
    word2id = dict([(w, j) for j, w in enumerate(vocab)])
    id2word = dict([(j, w) for j, w in enumerate(vocab)])
    print('  vocabulary after removing words not in train: {}'.format(len(vocab)))

    # split in train/test/valid
    docs_tr = [[word2id[w] for w in init_docs[idx_permute[idx_d]].split() if w in word2id] for idx_d in range(trSize)]
    docs_ts = [[word2id[w] for w in init_docs[idx_permute[idx_d+trSize]].split() if w in word2id] for idx_d in range(tsSize)]
    docs_va = [[word2id[w] for w in init_docs[idx_permute[idx_d+trSize+tsSize]].split() if w in word2id] for idx_d in range(vaSize)]

    print('  number of documents (train): {} [this should be equal to {}]'.format(len(docs_tr), trSize))
    print('  number of documents (test): {} [this should be equal to {}]'.format(len(docs_ts), tsSize))
    print('  number of documents (valid): {} [this should be equal to {}]'.format(len(docs_va), vaSize))

    # remove empty documents
    print('removing empty documents...')

    docs_tr = remove_empty(docs_tr)
    docs_ts = remove_empty(docs_ts)
    docs_va = remove_empty(docs_va)

    # remove test documents with length=1
    docs_ts = [doc for doc in docs_ts if len(doc)>1]

    # split test set in 2 halves 
    # this is required input for the document completion task
    print('splitting test documents in 2 halves...')
    docs_ts_h1 = [[w for i,w in enumerate(doc) if i<=len(doc)/2.0-1] for doc in docs_ts]
    docs_ts_h2 = [[w for i,w in enumerate(doc) if i>len(doc)/2.0-1] for doc in docs_ts]

    return word2id, {"docs_tr":docs_tr, "docs_ts":docs_ts, "docs_ts_h1": docs_ts_h1, "docs_ts_h2":docs_ts_h2, "docs_va":docs_va}

def _create_matrixes(word2id, docs_dict):
    bow_docs_dict = {}
    # getting lists of words and doc_indices
    for key, docs in docs_dict.items():
        print('creating lists of words_indices/doc_indices for {}'.format(key))
        word_indices = create_list_word_indices(docs)
        print('  len({}): {}'.format(key.replace("docs","words_indices"), len(word_indices)))
        doc_indices = create_doc_indices(docs)
        print('  len(np.unique({})): {} [this should be {}]'.format(key.replace("docs", "doc_indices"), len(np.unique(doc_indices)), len(docs)))
        n_docs = len(docs)
        print('creating bow representation for {}'.format(key))
        bow = create_bow(doc_indices, word_indices, n_docs, len(word2id))
        bow_docs_dict[key.replace("docs", "bow")] = bow
    return bow_docs_dict

def _export_matrixes(path_save, bow_docs_dict):
    if not os.path.isdir(path_save):
        os.system('mkdir -p ' + path_save)

    # Split bow intro token/value pairs
    print('splitting bow into token/value pairs and saving to disk...')
    for key, bow in bow_docs_dict.items():
        bow_tokens, bow_counts = split_bow(bow, bow.shape[0])
        savemat(path_save + key + '_tokens', {'tokens': bow_tokens}, do_compression=True)
        savemat(path_save + key + '_counts', {'counts': bow_counts}, do_compression=True)

def format_from_text(path_text, path_save, norm=False, max_df=0.7, min_df=10, ratio={"training":0.85, "testing": 0.1}):

    # read docs from file
    with open(path_text, 'r') as f:
        docs = f.readlines()
    
    if norm == True:
        # normalize textual content from docs
        docs = _normalize_txt(docs)
    
    # initialize a vocabulary dictionary
    word2id = _initialize_vocabulary_dict(docs, max_df, min_df)

    # split docs into training, validation, and testing set
    refined_word2id, docs_dict = _split_data(docs, word2id, ratio)

    # transform dataset into bag-of-words (bow) matrix
    bow_docs_dict = _create_matrixes(refined_word2id, docs_dict)

    # save bow to disk
    _export_matrixes(path_save, bow_docs_dict)

    # save vocabulary to disk
    with open(path_save + 'vocab.pkl', 'wb') as f:
        pickle.dump(refined_word2id, f)