from .data import get_batch

import math
import re
import pickle

import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np 

class ETM(nn.Module):
    def __init__(self, num_topics, vocab_size, t_hidden_size, rho_size, emsize, 
                    theta_act, embeddings=None, train_embeddings=True, enc_drop=0.5):
        super(ETM, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Current device is {}".format(self.device))

        ## define hyperparameters
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.t_hidden_size = t_hidden_size
        self.rho_size = rho_size
        self.enc_drop = enc_drop
        self.emsize = emsize
        self.t_drop = nn.Dropout(enc_drop)

        self.theta_act = self.get_activation(theta_act)
        
        ## define the word embedding matrix \rho
        if train_embeddings:
            self.rho = nn.Linear(rho_size, vocab_size, bias=False)
        else:
            self.rho = embeddings.clone().float().to(self.device)

        ## define the matrix containing the topic embeddings
        self.alphas = nn.Linear(rho_size, num_topics, bias=False)
    
        ## define variational distribution for \theta_{1:D} via amortizartion
        print("THE Vocabulary size is {}\n".format(vocab_size))
        self.q_theta = nn.Sequential(
                nn.Linear(vocab_size, t_hidden_size), 
                self.theta_act,
                nn.Linear(t_hidden_size, t_hidden_size),
                self.theta_act,
            )
        self.mu_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)
    
    def get_activation(self, act):
        """ return a activation function from nn module
        """
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        elif act == 'softplus':
            act = nn.Softplus()
        elif act == 'rrelu':
            act = nn.RReLU()
        elif act == 'leakyrelu':
            act = nn.LeakyReLU()
        elif act == 'elu':
            act = nn.ELU()
        elif act == 'selu':
            act = nn.SELU()
        elif act == 'glu':
            act = nn.GLU()
        else:
            print('Defaulting to tanh activations...')
            act = nn.Tanh()
        return act
    
    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar) 
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu
    
    def encode(self, bows):
        """Returns paramters of the variational distribution for \theta.
        input: bows
                batch of bag-of-words...tensor of shape bsz x V
        output: mu_theta, log_sigma_theta
        """
        q_theta = self.q_theta(bows)
        if self.enc_drop > 0:
            q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)
        kl_theta = -0.5 * torch.sum(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1).mean()
        return mu_theta, logsigma_theta, kl_theta
    
    def get_beta(self):
        """
        This generate the description as a defintion over words
        Returns:
            [type]: [description]
        """
        try:
            logit = self.alphas(self.rho.weight) # torch.mm(self.rho, self.alphas)
        except:
            logit = self.alphas(self.rho)
        beta = F.softmax(logit, dim=0).transpose(1, 0) ## softmax over vocab dimension
        return beta

    def get_theta(self, normalized_bows):
        """
        getting the topic poportion for the document passed in the normalixe bow
        """
        mu_theta, logsigma_theta, kld_theta = self.encode(normalized_bows)
        z = self.reparameterize(mu_theta, logsigma_theta)
        # z is 2-dimensional matrix (row: batch, column: hidden_dims)
        # dim=-1 is softmax on column-wise (normalize hidden_dims so that sum(z) = 1)
        theta = F.softmax(z, dim=-1) 
        return theta, kld_theta
    
    def decode(self, theta, beta):
        """compute the probability of topic given the document which is equal to theta^T ** B
        """
        res = torch.mm(theta, beta)
        almost_zeros = torch.full_like(res, 1e-6)
        results_without_zeros = res.add(almost_zeros)
        # value of prediction = good (0) ~ bad (-infinity)
        predictions = torch.log(results_without_zeros)
        return predictions

    def forward(self, bows, normalized_bows, theta=None, aggregate=True):
        ## get \theta
        if theta is None:
            theta, kld_theta = self.get_theta(normalized_bows)
        else:
            kld_theta = None

        ## get \beta
        beta = self.get_beta()

        ## get prediction loss
        preds = self.decode(theta, beta)
        # the imagination of calculating recon_loss
        # when prediction is good (0) that times a large count (N), resulting into a 0 loss
        # or prediction is bad (-infinity) that times a large count (N), resulting into a huge sloss
        recon_loss = -(preds * bows).sum(1)
        if aggregate:
            recon_loss = recon_loss.mean()
        return recon_loss, kld_theta

    def get_optimizer(self, args):
        """
        Get the model default optimizer 
        Args:
            sefl ([type]): [description]
        """
        if args.optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.wdecay)
        elif args.optimizer == 'adagrad':
            optimizer = optim.Adagrad(self.parameters(), lr=args.lr, weight_decay=args.wdecay)
        elif args.optimizer == 'adadelta':
            optimizer = optim.Adadelta(self.parameters(), lr=args.lr, weight_decay=args.wdecay)
        elif args.optimizer == 'rmsprop':
            optimizer = optim.RMSprop(self.parameters(), lr=args.lr, weight_decay=args.wdecay)
        elif args.optimizer == 'asgd':
            optimizer = optim.ASGD(self.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
        else:
            print('Defaulting to vanilla SGD')
            optimizer = optim.SGD(self.parameters(), lr=args.lr)
        self.optimizer = optimizer
        return optimizer
    
    def train_for_epoch(self, epoch, args, tokens, counts):
        """
        train the model for the given epoch 
        Args:
            epoch ([type]): [description]
        """
        self.train()
        acc_loss = 0
        acc_kl_theta_loss = 0
        cnt = 0
        perm_indices = torch.randperm(args.num_docs_train)
        batch_indices = torch.split(perm_indices, args.batch_size)
        print("The number of the indices I am using for the training is ", len(batch_indices))
        for idx, indices in enumerate(batch_indices):
            self.optimizer.zero_grad()
            self.zero_grad() 
            data_batch = get_batch(tokens, counts, indices, self.vocab_size, self.device)
            
            # normalize each data_batch so that the sum of each data_batch will equal to 1
            # the normalization help generator since the scale is more smooth
            sums = data_batch.sum(1).unsqueeze(1)
            if args.bow_norm:
                normalized_data_batch = data_batch / sums
            else:
                normalized_data_batch = data_batch
        
            recon_loss, kld_theta = self.forward(data_batch, normalized_data_batch)
            total_loss = recon_loss + kld_theta
            total_loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), args.clip)
            self.optimizer.step()

            acc_loss += torch.sum(recon_loss).item()
            acc_kl_theta_loss += torch.sum(kld_theta).item()
            cnt += 1
            if idx % args.log_interval == 0 and idx > 0:
                cur_loss = round(acc_loss / cnt, 2) 
                cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
                cur_real_loss = round(cur_loss + cur_kl_theta, 2)

                print('Epoch: {} .. batch: {}/{} .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
                    epoch, idx, math.ceil(args.num_docs_train/args.batch_size)-1, self.optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_real_loss))

        cur_loss = round(acc_loss / cnt, 2) 
        cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
        cur_real_loss = round(cur_loss + cur_kl_theta, 2)
        print('*'*100)
        print('Epoch----->{} .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
                epoch, self.optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_real_loss))
        print('*'*100)

    def evaluate(self, args, source, test_tokens_h1, test_counts_h1, test_tokens_h2, test_counts_h2):
        """
        Compute perplexity on document completion.
        """
        self.eval()
        with torch.no_grad():

            ## get \beta here
            beta = self.get_beta()

            ### do dc and tc here
            acc_loss = 0
            cnt = 0
            indices_1 = torch.split(torch.tensor(range(args.num_docs_test_1)), args.eval_batch_size)
            for _, indice in enumerate(indices_1): 
                data_batch_1 = get_batch(test_tokens_h1, test_counts_h1, indice, self.vocab_size, self.device)
                sums_1 = data_batch_1.sum(1).unsqueeze(1)
                if args.bow_norm:
                    normalized_data_batch_1 = data_batch_1 / sums_1
                else:
                    normalized_data_batch_1 = data_batch_1
                theta, _ = self.get_theta(normalized_data_batch_1)
                ## get predition loss using second half
                data_batch_2 = get_batch(test_tokens_h2, test_counts_h2, indice, self.vocab_size, self.device)
                sums_2 = data_batch_2.sum(1).unsqueeze(1)
                res = torch.mm(theta, beta)
                preds = torch.log(res)
                recon_loss = -(preds * data_batch_2).sum(1)

                loss = recon_loss / sums_2.squeeze()
                loss = np.nanmean(loss.cpu()) # old: loss.numpy()
                acc_loss += loss
                cnt += 1
            cur_loss = acc_loss / cnt
            ppl_dc = round(math.exp(cur_loss), 1)
            print('*'*100)
            print('{} Doc Completion PPL: {}'.format(source.upper(), ppl_dc))
            # print('*'*100)
            # if tc or td:
            #     beta = beta.data.cpu().numpy()
            #     if tc:
            #         print('Computing topic coherence...')
            #         get_topic_coherence(beta, training_set, vocabulary)
            #     if td:
            #         print('Computing topic diversity...')
            #         get_topic_diversity(beta, 25)
            return ppl_dc
    
    def get_topic_words(self, args, vocab):
        self.eval()
        with torch.no_grad():

            beta = self.get_beta()

            source_topics = dict()  
            target_topics = dict()
            words_all = list()

            for k in range(self.num_topics):#topic_indices:
                source_topics['topic_{}'.format(k)] = list()
                target_topics['topic_{}'.format(k)] = list()
                gamma = beta[k]
                # return top word index => https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
                top_words = list(gamma.cpu().numpy().argsort()[-args.num_words+1:][::-1])
                topic_words = [vocab[a] for a in top_words]
                words_all.append(topic_words)
                for word in topic_words:
                    if re.match("^[a-zA-Z].", word):
                        source_topics['topic_{}'.format(k)].append(word)
                    else:
                        target_topics['topic_{}'.format(k)].append(word)
        
        return words_all, source_topics, target_topics

    def infer_theta(self, args, tokens, counts):
        self.eval()
        with torch.no_grad():
            assert len(tokens) == len(counts)
            num_docs = len(tokens)
            indices = torch.tensor(range(num_docs))
            indices = torch.split(indices, args.eval_batch_size)

            thetaList = list()

            for _, ind in enumerate(indices):
                data_batch = get_batch(tokens, counts, ind, self.vocab_size, self.device)
                sums = data_batch.sum(1).unsqueeze(1)

                if args.bow_norm:
                    normalized_data_batch = data_batch / sums
                else:
                    normalized_data_batch = data_batch

                theta, _ = self.get_theta(normalized_data_batch)
                thetaList.append(theta)

        return torch.cat(tuple(thetaList), 0)
