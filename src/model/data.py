import numpy as np
import torch 
import os
import pickle
from scipy.io import loadmat

def _fetch(path, name):
    
    # concatenate the file path
    if name == 'train':
        token_file = os.path.join(path, 'bow_tr_tokens')
        count_file = os.path.join(path, 'bow_tr_counts')
    # elif name == 'full':
    #     token_file = os.path.join(path, 'bow_full_tokens.mat')
    #     count_file = os.path.join(path, 'bow_full_counts.mat')
    elif name == 'valid':
        token_file = os.path.join(path, 'bow_va_tokens')
        count_file = os.path.join(path, 'bow_va_counts')
    elif name == 'test':
         token_file = os.path.join(path, 'bow_ts_tokens')
         count_file = os.path.join(path, 'bow_ts_counts')

    # load data matrix using file path
    tokens = loadmat(token_file)['tokens'].squeeze()
    counts = loadmat(count_file)['counts'].squeeze()

    if name == 'test':
        token_1_file = os.path.join(path, 'bow_ts_h1_tokens')
        count_1_file = os.path.join(path, 'bow_ts_h1_counts')
        token_2_file = os.path.join(path, 'bow_ts_h2_tokens')
        count_2_file = os.path.join(path, 'bow_ts_h2_counts')
        tokens_1 = loadmat(token_1_file)['tokens'].squeeze()
        counts_1 = loadmat(count_1_file)['counts'].squeeze()
        tokens_2 = loadmat(token_2_file)['tokens'].squeeze()
        counts_2 = loadmat(count_2_file)['counts'].squeeze()
        return {'tokens': tokens, 'counts': counts, 
                    'tokens_1': tokens_1, 'counts_1': counts_1, 
                        'tokens_2': tokens_2, 'counts_2': counts_2}

    return {'tokens': tokens, 'counts': counts}

def get_data(path):
    # load vocabulary dictionary
    with open(os.path.join(path, 'vocab.pkl'), 'rb')as f:
        vocab = pickle.load(f)

    train = _fetch(path, 'train')
    valid = _fetch(path, 'valid')
    test = _fetch(path, 'test')
    # full = _fetch(path, 'full')

    return vocab, train, valid, test

def get_batch(tokens, counts, indices, vocab_size, device):
    """fetch input data by batch."""
    batch_size = len(indices)
    data_batch = np.zeros((batch_size, vocab_size))
    
    for i, doc_id in enumerate(indices):
        # select data
        doc = tokens[doc_id]
        count = counts[doc_id]

        if len(doc) == 1: 
            doc = [doc.squeeze()]
            count = [count.squeeze()]
        else:
            doc = doc.squeeze()
            count = count.squeeze()

        if doc_id != -1:
            for j, word in enumerate(doc):
                if word.size == 0:
                    continue
                data_batch[i, word] = count[j]

    data_batch = torch.from_numpy(data_batch).float().to(device)
    return data_batch

# Class for a memory-friendly iterator over the dataset
class MemoryFriendlyFileIterator(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in open(self.filename):
            yield line.strip().split()

def _get_extension(path):
    assert isinstance(path, str), 'path extension is not str'
    filename = path.split(os.path.sep)[-1]
    return filename.split('.')[-1]

def embedding_reader(emb_path, vocab, emb_size):
    assert _get_extension(emb_path) == 'txt', "the embedding file should be in txt file."
    
    # read word vectors from txt file
    iterator = MemoryFriendlyFileIterator(emb_path)
    vectors = {}
    for line in iterator:
        word = line[0]
        if word in vocab:
            vect = np.array(line[1:]).astype(np.float)
            vectors[word] = vect
    
    # reorder vectors and construct embedding matrix
    model_embeddings = np.zeros((len(vocab), emb_size))

    not_found = 0
    for word, index in vocab.items():
        try:
            model_embeddings[index] = vectors[word]
        except KeyError:
            not_found += 1
            model_embeddings[index] = np.random.normal(
                scale=0.6, size=(emb_size, ))
    print("{}({}) words are not found.".format(not_found, round(not_found/len(vocab), 2)))
    return model_embeddings