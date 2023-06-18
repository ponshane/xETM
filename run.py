import argparse
import json
import os
from collections import Counter
from datetime import datetime
from itertools import chain
from pprint import pprint

import numpy as np
import pandas as pd
import torch
from scipy.spatial import distance
from scipy.stats import entropy

from src.evaluate import Evaluator
from src.model.data import embedding_reader, get_data
from src.model.etm import ETM

parser = argparse.ArgumentParser(description="Cross-lingual Embedded Topic Model")

### data and file related arguments
parser.add_argument("--data_path", type=str, help="directory containing data")
parser.add_argument("--emb_path", type=str, help="directory containing word embeddings")
parser.add_argument(
    "--save_path", type=str, default="./results", help="path to save results"
)
parser.add_argument(
    "--batch_size", type=int, default=256, help="input batch size for training"
)
parser.add_argument(
    "--suffix", type=str, default="", help="suffix of eval production"
)  # just don't set default. source: https://stackoverflow.com/questions/38533258/how-can-argparse-set-default-value-of-optional-parameter-to-null-or-empty
parser.add_argument(
    "--data_label_path", type=str, help="the label of the training data"
)
parser.add_argument("--shuffle_data_order_path", type=str, help="shuffle data order")
parser.add_argument("--full_data_path", type=str, help="the path of the training data")


### model-related arguments
parser.add_argument(
    "--train_embeddings", type=int, default=0, help="whether to fix rho or train it"
)
parser.add_argument("--num_topics", type=int, default=50, help="number of topics")
parser.add_argument("--rho_size", type=int, default=300, help="dimension of rho")
parser.add_argument("--emb_size", type=int, default=300, help="dimension of embeddings")
parser.add_argument(
    "--t_hidden_size",
    type=int,
    default=800,
    help="dimension of hidden space of q(theta)",
)
parser.add_argument(
    "--theta_act",
    type=str,
    default="relu",
    help="tanh, softplus, relu, rrelu, leakyrelu, elu, selu, glu)",
)

### optimization-related arguments
parser.add_argument("--bow_norm", type=int, default=1, help="normalize the bows or not")
parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
parser.add_argument("--epochs", type=int, default=20, help="number of epochs to train")
parser.add_argument("--mode", type=str, default="train", help="train or eval model")
parser.add_argument("--optimizer", type=str, default="adam", help="choice of optimizer")
parser.add_argument("--seed", type=int, default=168, help="random seed (default: 168)")
parser.add_argument(
    "--enc_drop", type=float, default=0.0, help="dropout rate on encoder"
)
parser.add_argument(
    "--wdecay", type=float, default=1.2e-6, help="some l2 regularization"
)
parser.add_argument("--clip", type=float, default=0.0, help="gradient clipping")

### anneal-related arguments (false as default)
parser.add_argument(
    "--anneal_lr",
    type=int,
    default=0,
    help="whether to anneal the learning rate or not",
)
parser.add_argument(
    "--nonmono", type=int, default=10, help="number of bad hits allowed"
)
parser.add_argument(
    "--lr_factor", type=float, default=4.0, help="divide learning rate by this..."
)

### evaluation, visualization, and logging-related arguments
parser.add_argument("--log_interval", type=int, default=10, help="when to log training")
# parser.add_argument('--visualize_every', type=int, default=10, help='when to visualize results')
parser.add_argument(
    "--num_words", type=int, default=10000, help="number of words for topic viz"
)
parser.add_argument(
    "--eval_batch_size", type=int, default=256, help="input batch size for evaluation"
)
parser.add_argument(
    "--load_from", type=str, default="", help="the name of the ckpt to eval from"
)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n")
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
train_tokens = train["tokens"]
train_counts = train["counts"]
args.num_docs_train = len(train_tokens)

# 2. dev set
# valid_tokens = valid['tokens']
# valid_counts = valid['counts']
# args.num_docs_valid = len(valid_tokens)

# 3. test data
test_tokens = test["tokens"]
test_counts = test["counts"]
args.num_docs_test = len(test_tokens)
test_1_tokens = test["tokens_1"]
test_1_counts = test["counts_1"]
args.num_docs_test_1 = len(test_1_tokens)
test_2_tokens = test["tokens_2"]
test_2_counts = test["counts_2"]
args.num_docs_test_2 = len(test_2_tokens)


embeddings = None
if not args.train_embeddings:
    embeddings = embedding_reader(args.emb_path, vocab, args.emb_size)
    embeddings = torch.from_numpy(embeddings).to(device)

""" define checkpoint
"""
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

if args.mode == "eval":
    ckpt = args.load_from
else:
    ckpt = os.path.join(
        args.save_path,
        "D_{}_K_{}_Epo_{}_Opt_{}".format(
            args.emb_size, args.num_topics, args.epochs, args.optimizer
        ),
    )

""" initialize model and optimizer
"""
model = ETM(
    args.num_topics,
    vocab_size,
    args.t_hidden_size,
    args.rho_size,
    args.emb_size,
    args.theta_act,
    embeddings,
    args.train_embeddings,
    args.enc_drop,
).to(device)

print("model: {}".format(model))

optimizer = model.get_optimizer(args)

if args.mode == "train":
    """train model on data"""
    best_epoch = 0
    best_val_ppl = 1e9
    all_val_ppls = []

    print("\n")
    for epoch in range(0, args.epochs):
        print(datetime.now())
        print("I am training for epoch", epoch)
        model.train_for_epoch(epoch, args, train_tokens, train_counts)
        val_ppl = model.evaluate(
            args, "val", test_1_tokens, test_1_counts, test_2_tokens, test_2_counts
        )
        print("The validation scores", val_ppl)

        if val_ppl < best_val_ppl:
            # with open(ckpt, 'wb') as f:
            #     torch.save(model, f)
            best_epoch = epoch
            best_val_ppl = val_ppl
        else:
            ## check whether to anneal lr
            lr = optimizer.param_groups[0]["lr"]
            if args.anneal_lr and (
                len(all_val_ppls) > args.nonmono
                and val_ppl > min(all_val_ppls[: -args.nonmono])
                and lr > 1e-5
            ):
                optimizer.param_groups[0]["lr"] /= args.lr_factor

        all_val_ppls.append(val_ppl)

    with open(ckpt, "wb") as f:
        torch.save(model, f)
    with open(ckpt, "rb") as f:
        model = torch.load(f)

    model = model.to(device)
    print(datetime.now())
    print(
        "Epoch {} has the best model whose PPL is {}".format(best_epoch, best_val_ppl)
    )
    val_ppl = model.evaluate(
        args, "val", test_1_tokens, test_1_counts, test_2_tokens, test_2_counts
    )

elif args.mode == "eval":
    with open(ckpt, "rb") as f:
        model = torch.load(f)
    model = model.to(device)

    reverse_vocab = {v: k for k, v in vocab.items()}
    words_all, source_topics, target_topics = model.get_topic_words(args, reverse_vocab)

    ## initial evaluator
    evaluator = Evaluator(args, vocab, model, words_all)
    evaluator.inference_all()

    ## c-npmi

    # save topic words
    with open(
        f"./experiment/topic_word/topic_words_D_{args.emb_size}_K_{args.num_topics}_{args.suffix}.txt",
        "w",
    ) as f:
        for words in words_all:
            f.write(" ".join(words) + "\n")
    # save source topic
    with open(
        f"./experiment/topic_word/source_topic_D_{args.emb_size}_K_{args.num_topics}_{args.suffix}.json",
        "w",
    ) as f:
        json.dump(source_topics, f, indent=4)

    # save target topic
    with open(
        f"./experiment/topic_word/target_topic_D_{args.emb_size}_K_{args.num_topics}_{args.suffix}.json",
        "w",
    ) as f:
        json.dump(target_topics, f, indent=4, ensure_ascii=False)
