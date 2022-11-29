import string
from scipy import sparse

""" string processing
"""
def contains_punctuation(w):
    return any(char in string.punctuation for char in w)

def contains_numeric(w):
    return any(char.isdigit() for char in w)

""" processing for list of documents
"""
def remove_empty(in_docs):
    return [doc for doc in in_docs if doc!=[]]

def create_list_word_indices(in_docs):
    return [word for doc in in_docs for word in doc]

def create_doc_indices(in_docs):
    # j is the document index
    # replicate j by len(doc) times
    aux = [[j for i in range(len(doc))] for j, doc in enumerate(in_docs)]
    return [int(index) for index_list in aux for index in index_list]

def create_bow(doc_indices, word_indices, n_docs, vocab_size):
    """ this function helps build document-term matrix in a scipy.sparse matrix format
    API Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html

    for simplicity,
        coo_matrix((data, (i, j)), [shape=(M, N)]) generates A
        where  A[i[k], j[k]] = data[k]
    note,
        COO is a fast format for constructing sparse matrices
        Once a matrix has been constructed, convert to CSR or CSC format for fast arithmetic and matrix vector operations
        by default when converting to CSR or CSC format, duplicate (i,j) entries will be summed together
    """
    return sparse.coo_matrix(([1]*len(doc_indices),(doc_indices, word_indices)), shape=(n_docs, vocab_size)).tocsr()

def split_bow(bow_in, n_docs):
        indices = [[w for w in bow_in[doc,:].indices] for doc in range(n_docs)]
        counts = [[c for c in bow_in[doc,:].data] for doc in range(n_docs)]
        return indices, counts