import argparse
import os
import json
from itertools import chain
from collections import Counter
from pprint import pprint

import torch
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.stats import entropy

from src.model.data import get_data, embedding_reader
from src.model.etm import ETM
from src.utils import compute_purity

parser = argparse.ArgumentParser(description='Cross-lingual Embedded Topic Model')

### data and file related arguments
parser.add_argument('--data_path', type=str, help='directory containing data')
parser.add_argument('--emb_path', type=str, help='directory containing word embeddings')
parser.add_argument('--save_path', type=str, default='./results', help='path to save results')
parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training')
parser.add_argument('--suffix', type=str, default="", help='suffix of eval production') # just don't set default. source: https://stackoverflow.com/questions/38533258/how-can-argparse-set-default-value-of-optional-parameter-to-null-or-empty
parser.add_argument('--data_label_path', type=str, help='the label of the traiing data')
parser.add_argument('--shuffle_data_order_path', type=str, help='the label of the traiing data')


### model-related arguments
parser.add_argument('--train_embeddings', type=int, default=0, help='whether to fix rho or train it')
parser.add_argument('--num_topics', type=int, default=50, help='number of topics')
parser.add_argument('--rho_size', type=int, default=300, help='dimension of rho')
parser.add_argument('--emb_size', type=int, default=300, help='dimension of embeddings')
parser.add_argument('--t_hidden_size', type=int, default=800, help='dimension of hidden space of q(theta)')
parser.add_argument('--theta_act', type=str, default='relu', help='tanh, softplus, relu, rrelu, leakyrelu, elu, selu, glu)')

### optimization-related arguments
parser.add_argument('--bow_norm', type=int, default=1, help='normalize the bows or not')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
parser.add_argument('--mode', type=str, default='train', help='train or eval model')
parser.add_argument('--optimizer', type=str, default='adam', help='choice of optimizer')
parser.add_argument('--seed', type=int, default=168, help='random seed (default: 168)')
parser.add_argument('--enc_drop', type=float, default=0.0, help='dropout rate on encoder')
parser.add_argument('--wdecay', type=float, default=1.2e-6, help='some l2 regularization')
parser.add_argument('--clip', type=float, default=0.0, help='gradient clipping')

### anneal-related arguments (false as default)
parser.add_argument('--anneal_lr', type=int, default=0, help='whether to anneal the learning rate or not')
parser.add_argument('--nonmono', type=int, default=10, help='number of bad hits allowed')
parser.add_argument('--lr_factor', type=float, default=4.0, help='divide learning rate by this...')

### evaluation, visualization, and logging-related arguments
parser.add_argument('--log_interval', type=int, default=10, help='when to log training')
# parser.add_argument('--visualize_every', type=int, default=10, help='when to visualize results')
parser.add_argument('--num_words', type=int, default=10000, help='number of words for topic viz')
parser.add_argument('--eval_batch_size', type=int, default=256, help='input batch size for evaluation')
parser.add_argument('--load_from', type=str, default='', help='the name of the ckpt to eval from')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('\n')
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

""" retrieve formatted dataset and vocabulary
"""
# 0. vocabulary
vocab, train, valid, test = get_data(os.path.join(args.data_path))
vocab_size = len(vocab)

# 1. training data
train_tokens = train['tokens']
train_counts = train['counts']
args.num_docs_train = len(train_tokens)

# 2. dev set
# valid_tokens = valid['tokens']
# valid_counts = valid['counts']
# args.num_docs_valid = len(valid_tokens)

# 3. test data
test_tokens = test['tokens']
test_counts = test['counts']
args.num_docs_test = len(test_tokens)
test_1_tokens = test['tokens_1']
test_1_counts = test['counts_1']
args.num_docs_test_1 = len(test_1_tokens)
test_2_tokens = test['tokens_2']
test_2_counts = test['counts_2']
args.num_docs_test_2 = len(test_2_tokens)


embeddings = None
if not args.train_embeddings:
    embeddings = embedding_reader(args.emb_path, vocab, args.emb_size)
    embeddings = torch.from_numpy(embeddings).to(device)

""" define checkpoint
""" 
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

if args.mode == 'eval':
    ckpt = args.load_from
else:
    ckpt = os.path.join(args.save_path, 
        'D_{}_K_{}_Epo_{}_Opt_{}'.format(
        args.emb_size, args.num_topics, args.epochs, args.optimizer))

""" initialize model and optimizer
"""
model = ETM(args.num_topics, 
            vocab_size, 
            args.t_hidden_size, 
            args.rho_size, 
            args.emb_size, 
            args.theta_act, 
            embeddings, 
            args.train_embeddings, 
            args.enc_drop).to(device)

print('model: {}'.format(model))

optimizer = model.get_optimizer(args)

if args.mode == 'train':

    """ train model on data
    """  
    best_epoch = 0
    best_val_ppl = 1e9
    all_val_ppls = []

    print('\n')
    for epoch in range(0, args.epochs):

        print("I am training for epoch", epoch)
        model.train_for_epoch(epoch, args, train_tokens, train_counts)
        val_ppl = model.evaluate(args, 'val', test_1_tokens, test_1_counts, test_2_tokens, test_2_counts)
        print("The validation scores", val_ppl)

        if val_ppl < best_val_ppl:
            # with open(ckpt, 'wb') as f:
            #     torch.save(model, f)
            best_epoch = epoch
            best_val_ppl = val_ppl
        else:
            ## check whether to anneal lr
            lr = optimizer.param_groups[0]['lr']
            if args.anneal_lr and (len(all_val_ppls) > args.nonmono and val_ppl > min(all_val_ppls[:-args.nonmono]) and lr > 1e-5):
                optimizer.param_groups[0]['lr'] /= args.lr_factor
        
        all_val_ppls.append(val_ppl)
        
    with open(ckpt, 'wb') as f:
        torch.save(model, f)
    with open(ckpt, 'rb') as f:
        model = torch.load(f)

    model = model.to(device)
    print("Epoch {} has the best model whose PPL is {}".format(best_epoch, best_val_ppl))
    val_ppl = model.evaluate(args, 'val', test_1_tokens, test_1_counts, test_2_tokens, test_2_counts)

elif args.mode == "eval":
    with open(ckpt, 'rb') as f:
        model = torch.load(f)
    model = model.to(device)

    reverse_vocab = {v: k for k, v in vocab.items()}
    words_all, source_topics, target_topics = model.get_topic_words(args, reverse_vocab)

    ## load full docs
    from src.preprocessing.format_txt_into_mat import _create_matrixes, split_bow
    # TODO: make full corpus to id
    with open("./data/training_all_6011.txt", "r") as f:
        docs = [line.strip().split() for line in f.readlines()]
        docs = [list(filter(lambda x: x is not None, map(vocab.get, doc))) for doc in docs]


    bow_dict = _create_matrixes(vocab, {"full": docs})
    bow = bow_dict['full']
    bow_tokens, bow_counts = split_bow(bow, bow.shape[0])
    
    token_arr = [np.array(doc).reshape(1,len(doc)) for doc in bow_tokens]
    tmp_arr = np.empty(len(token_arr), object)
    tmp_arr[:] = token_arr
    full_tokens = tmp_arr

    count_arr = [np.array(doc).reshape(1,len(doc)) for doc in bow_counts]
    tmp_arr = np.empty(len(count_arr), object)
    tmp_arr[:] = count_arr
    full_counts = tmp_arr
    ## try infer from full corpus
    thetas = model.infer_theta(args, full_tokens, full_counts)

    # thetas = model.infer_theta(args, train_tokens, train_counts)
    thetas = thetas.cpu().numpy()
    print(thetas[0], thetas[0].shape)
    print(thetas.shape)
    # save thetas 
    # np.save(
    #     f"./experiment/thetas_K_{args.num_topics}", 
    #     thetas.cpu().numpy()
    # )

    ## remove overlap words in top 200 
    n = 200
    print(f"diversity {n}: {len(np.unique([word[:n] for word in words_all])) / (n*50)}")
    inter_words_cnt = Counter(chain(*[word[:n] for word in words_all]))
    inter_total_cnt = 0
    for w, count in inter_words_cnt.items():
        if count > 1:
            inter_total_cnt += count
    print(f"inter of {n}: {inter_total_cnt/(n*50)}")
    with open("./intersect.txt", "w") as f:
        for w in inter_words_cnt.most_common():
            f.write(f"{w}\n")
    inter_words = [word for word, count in inter_words_cnt.items() if count > 1]

    for word in words_all:
        unique_words = [w for w in word[:n] if w not in inter_words]
        print(unique_words)

    
    # print(source_topics, target_topics)
    
    # save topic words
    with open(f"./experiment/topic_word/topic_words_D_{args.emb_size}_K_{args.num_topics}_{args.suffix}.txt", "w") as f:
        for words in words_all:
            f.write(" ".join(words) + "\n")
    # save source topic
    with open(f"./experiment/topic_word/source_topic_D_{args.emb_size}_K_{args.num_topics}_{args.suffix}.json", "w") as f:
        json.dump(source_topics, f, indent=4)
    
    # save target topic 
    with open(f"./experiment/topic_word/target_topic_D_{args.emb_size}_K_{args.num_topics}_{args.suffix}.json",
     "w"
    ) as f:
        json.dump(target_topics, f, indent=4, ensure_ascii=False)

    print(f"entropy of theta:")
    print(pd.Series(entropy(thetas, axis=1)).describe())
    print(f"average topic share:")
    print(np.nanmean(thetas, axis=0))

    # doc_order = pd.read_csv(args.shuffle_data_order_path, header=None) ## need to adjust following training data
    # doc_order.columns=["shuffle_order"]
    ## original docs need to arrange in `shuffle order`
    with open(args.shuffle_data_order_path, "r") as f:
        origin_position = [l.strip() for l in f.readlines()]
    doc_order = pd.DataFrame({"origin_position": origin_position})
    doc_order["now_position"] = list(doc_order.index)
    doc_order = doc_order.astype(int)

    ## compute the jsd
    remain_arts = doc_order.iloc[:thetas.shape[0], :] #training docs
    total_len = doc_order.shape[0]
    en_origin_index = np.arange(0, total_len/2,dtype=int)
    # zh_origin_index = np.arange(total_len/2, total_len, dtype=int)
    zh_origin_index = en_origin_index + int(total_len/2)
    # 找出 origin idx 還剩多少在現在的 training data
    zh_en_idx = np.in1d(zh_origin_index, remain_arts.origin_position.values) & np.in1d(en_origin_index, remain_arts.origin_position.values)
    en_remain_idx = remain_arts.origin_position.isin(en_origin_index[zh_en_idx]).values
    zh_remain_idx = remain_arts.origin_position.isin(zh_origin_index[zh_en_idx]).values
    en_now_position_idx = remain_arts.loc[en_remain_idx,:].sort_values("origin_position")['now_position'].values
    zh_now_position_idx = remain_arts.loc[zh_remain_idx,:].sort_values("origin_position")['now_position'].values
    # zh_shuffle_idx = doc_order.loc[zh_index[zh_en_idx],'shuffle_order'].values
    # en_shuffle_idx = doc_order.loc[en_index[zh_en_idx],'shuffle_order'].values
    jsd = distance.jensenshannon(
        thetas[en_now_position_idx,:],
        thetas[zh_now_position_idx,:],
        axis = 1
    )
    
    
    jsd = distance.jensenshannon(
        thetas[np.arange(0, int(thetas.shape[0]/2)), :],
        thetas[np.arange(int(thetas.shape[0]/2), thetas.shape[0]), :],
        axis = 1
    )
    print("JSD:")
    print(pd.Series(jsd).describe())

    n = 15000
    print("the comparable thetas and docs:")
    # print(
    #     thetas[en_now_position_idx,:][n,:],
    #     "\n",
    #     thetas[zh_now_position_idx,:][n,:],
    # )
    print(
        thetas[n,:],
        "\n",
        thetas[int(n+thetas.shape[0]/2),:],
    )

    # awesome method https://stackoverflow.com/questions/16992713/translate-every-element-in-numpy-array-according-to-key/16992783#16992783
    # print(en_now_position_idx[n], zh_now_position_idx[n])
    # print(
    #     ", ".join(np.vectorize(reverse_vocab.get)(train_tokens[en_now_position_idx[n]][0])),
    #     "\n",
    #     ", ".join(np.vectorize(reverse_vocab.get)(train_tokens[zh_now_position_idx[n]][0])),
    # )
    # ", ".join(np.vectorize(reverse_vocab.get)(train_tokens[zh_now_position_idx[n]][0]))
    # remain_arts.loc[remain_arts.now_position == zh_now_position_idx[n],:]

    print(en_now_position_idx[n], zh_now_position_idx[n])
    print(
        ", ".join(np.vectorize(reverse_vocab.get)(full_tokens[n])[0]),
        "\n",
        ", ".join(np.vectorize(reverse_vocab.get)(full_tokens[int(n+thetas.shape[0]/2)])[0]),

    )

    ## the max agreement of each docs
    rows = np.arange(thetas.shape[0])
    print("the distribution max proportion topic of each docs")
    print(pd.DataFrame(thetas[rows, thetas.argmax(axis=1)]).describe().to_string())

    # compute purity
    with open(args.data_label_path) as f:
        college_label = [label.strip() for label in f.readlines()]
    
    ## reorder docs
    # shuffle_college_label=pd.Series(college_label).iloc[remain_arts.origin_position].values

    # df = pd.DataFrame({
    # "topic_labels": thetas.argmax(axis=1),
    # "college_labels": shuffle_college_label
    # })

    df = pd.DataFrame({
    "topic_labels": thetas.argmax(axis=1),
    "college_labels": college_label
    })
    print(compute_purity(df).join(df.topic_labels.value_counts().to_frame()).sort_values("topic_labels", ascending=False))
   
    print("Finish Theta Inference.")