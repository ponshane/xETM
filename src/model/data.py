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
                data_batch[i, word] = count[j]

    data_batch = torch.from_numpy(data_batch).float().to(device)
    return data_batch