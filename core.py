import torch
import torchvision 
from collections import defaultdict, Counter
import math 
from tqdm import tqdm 
from statsmodels.stats.proportion import proportion_confint
from datasets import get_dataset
import models 
from thresholds import get_thresholds, get_thresholds_union_bound
import numpy as np 
import heapq
from scipy.stats import norm 

tqdm = lambda x: x

class UniformNoise():
    def __init__(self, sigma):
        self.sigma = sigma 
    
    def get_noise_batch(self,x):
        device = x.get_device()
        return (torch.rand(x.shape, device=device)-0.5)*self.sigma*2 + x

class GaussianNoise():
    def __init__(self, sigma):
        self.sigma = sigma 
    
    def get_noise_batch(self,x):
        device = x.get_device()
        return torch.randn(x.shape, device=device) * self.sigma + x 

class Certifier():
    def __init__(self, sigma, noise):
        if(noise == 'gaussian'):
            self.noise = GaussianNoise(sigma)
        elif(noise == 'uniform'):
            self.noise = UniformNoise(sigma)
        else:
            raise ValueError("noise should be split or uniform, received " + str(noise))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_noise_batch = self.noise.get_noise_batch


    def certify(self, model, dataset, bs = 32, n0=128, n=100000, alpha=0.001, skip=20, targetp=0.8):
        model.eval()
        n0 = math.ceil(n0/bs)*bs 
        n = math.ceil(n/bs)*bs 
        dataset = get_dataset(dataset,'test') 
        ret = {}
        d1 = { }
        d2 = { }
        total = 0

        image_inds = range(0,len(dataset),skip)
        h = []
        cnts= {}
        for i in range(0,len(dataset),skip):
            heapq.heappush(h, (0, i))
            d1[i] = [] 
            cnts[i] = (0,0)
            total += 1

        X = torch.zeros((bs, *dataset[0][0].shape)).to(self.device)
        Y = torch.zeros((bs,))
        new = [0]*bs

        for i in range(bs):
            (cnt, idx) = heapq.heappop(h)
            d1[idx].append(i)
            d2[i] = idx
            x, y = dataset[idx]
            X[i] = x.to(self.device)
            Y[i] = y
            heapq.heappush(h, (cnt+1, idx))

        lower_threshold, upper_threshold = get_thresholds(p=targetp, alpha=alpha, n=131100)



        done = 0
        with torch.no_grad():
            while(done < total):
                smoothX = self.get_noise_batch(X)
                correct_preds = list(model(smoothX).argmax(-1).cpu() == Y)


                for i, correct in enumerate(correct_preds):
                    if(new[i] == 1):
                        new[i] = 0
                        continue 

                    idx = d2[i]
                    A, N = cnts[idx]
                    A +=  correct 
                    N += 1
                    cnts[idx] = (A,N)

                    if(N >= len(lower_threshold)):
                        ret[idx] = (-1, N)
                    elif(lower_threshold[N] == A):
                        ret[idx] = (0, N)
                    elif(upper_threshold[N] == A):
                        ret[idx] = (1, N)
                    else:
                        continue
                    
                    done += 1
                    if(done == total):
                        break

                    for j in d1[idx]:
                        new[j] = 1 

                        while(1):
                            (cnt, idx_) = heapq.heappop(h)
                            if(idx_ not in ret):
                                break

                        d1[idx_].append(j)
                        d2[j] = idx_
                        if(cnt == 0):
                            x, y = dataset[idx_]
                            X[j] = x.to(self.device)
                            Y[j] = y
                        else:
                            X[j] = X[d1[idx_][0]]
                            Y[j] = Y[d1[idx_][0]]
                        heapq.heappush(h, (cnt+1, idx_))
            return ret 


    def certify_ub(self, model, dataset, bs = 32, n0=128, n=100000, alpha=0.001, skip=20, targetp=0.8):
        model.eval()
        n0 = math.ceil(n0/bs)*bs 
        n = math.ceil(n/bs)*bs 
        dataset = get_dataset(dataset,'test') 
        ret = {}
        d1 = { }
        d2 = { }
        total = 0

        image_inds = range(0,len(dataset),skip)
        h = []
        cnts= {}
        for i in range(0,len(dataset),skip):
            heapq.heappush(h, (0, i))
            d1[i] = [] 
            cnts[i] = (0,0)
            total += 1

        X = torch.zeros((bs, *dataset[0][0].shape)).to(self.device)
        Y = torch.zeros((bs,))
        new = [0]*bs

        for i in range(bs):
            (cnt, idx) = heapq.heappop(h)
            d1[idx].append(i)
            d2[i] = idx
            x, y = dataset[idx]
            X[i] = x.to(self.device)
            Y[i] = y
            heapq.heappush(h, (cnt+1, idx))

        lower_threshold, upper_threshold = get_thresholds_union_bound(p=targetp, alpha=alpha, n=131100)



        done = 0
        with torch.no_grad():
            while(done < total):
                smoothX = self.get_noise_batch(X)
                correct_preds = list(model(smoothX).argmax(-1).cpu() == Y)


                for i, correct in enumerate(correct_preds):
                    if(new[i] == 1):
                        new[i] = 0
                        continue 

                    idx = d2[i]
                    A, N = cnts[idx]
                    A +=  correct 
                    N += 1
                    cnts[idx] = (A,N)

                    if(N >= len(lower_threshold)):
                        ret[idx] = (-1, N)
                    elif(lower_threshold[N] == A):
                        ret[idx] = (0, N)
                    elif(upper_threshold[N] == A):
                        ret[idx] = (1, N)
                    else:
                        continue
                    
                    done += 1
                    if(done == total):
                        break

                    for j in d1[idx]:
                        new[j] = 1 

                        while(1):
                            (cnt, idx_) = heapq.heappop(h)
                            if(idx_ not in ret):
                                break

                        d1[idx_].append(j)
                        d2[j] = idx_
                        if(cnt == 0):
                            x, y = dataset[idx_]
                            X[j] = x.to(self.device)
                            Y[j] = y
                        else:
                            X[j] = X[d1[idx_][0]]
                            Y[j] = Y[d1[idx_][0]]
                        heapq.heappush(h, (cnt+1, idx_))
            return ret 



    def certify_adaptive(self, model, dataset, bs = 32, n0=128, n=100000, alpha=0.001, skip=20, targetp=0.8):
        model.eval()
        dataset = get_dataset(dataset,'test') 

        ns = [100, 1000, 10000, 120000]
        ret = []

        with torch.no_grad():
            for idx in tqdm(range(0,len(dataset),skip)):
                X, y = dataset[idx]
                X = X.to(self.device)
                X = X.repeat((bs, 1, 1, 1))

                A = 0
                N = 0
                for n in ns:
                    for _ in range(n // bs):
                        x = self.get_noise_batch(X)
                        A +=  (model(x).argmax(-1) == y).sum().cpu()
                        N += bs 
                    else:
                        rem = n-(n//bs)*bs
                        x = self.get_noise_batch(X[:rem])
                        A +=  (model(x).argmax(-1) == y).sum().cpu()
                        N += rem


                    lo, hi = proportion_confint(A, N, alpha = alpha*2/len(ns), method='beta')
                    if(lo > targetp):
                        ret.append((1, N))
                        break
                    elif(hi < targetp):
                        ret.append((0, N))
                        break 
                else:
                    ret.append((-1, N))
                    
            return ret 


    def certify_vanilla(self, model, dataset, bs = 32, n0=128, n=10000, alpha=0.001, skip=20, targetp=0.8):
        model.to(self.device)
        model.eval()
        dataset = get_dataset(dataset,'test') 

        ret = []

        with torch.no_grad():
            for idx in tqdm(range(0,len(dataset),skip)):
                X, y = dataset[idx]
                X = X.to(self.device)
                X = X.repeat((bs, 1, 1, 1))

                A = 0
                N = 0
                for _ in range(n // bs):
                    x = self.get_noise_batch(X)
                    model(x)
                    A +=  (model(x).argmax(-1) == y).sum().cpu()
                    N += bs 
                
                ret.append((A,N))
            return ret 

class Trainer():
    def __init__(self, dataset, noise, lambd):

        if(noise == 'gaussian'):
            self.noise = GaussianNoise(lambd)
        elif(noise == 'uniform'):
            self.noise = UniformNoise(lambd)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_noise_batch = self.noise.get_noise_batch
        self.dataset = dataset 

        if(dataset == 'cifar'):
            self.model = models.WideResNet(dataset, self.device)
        elif(dataset == 'imagenet'):
            self.model = models.ResNet(dataset, self.device)

        self.model.train()

    def train(self, bs = 1000, lr = 0.1, num_epochs = 120, stability = False):
        train_loader = torch.utils.data.DataLoader(get_dataset(self.dataset, "train"),
                                shuffle=True,
                                batch_size=bs,
                                num_workers=2,
                                pin_memory=False)

        optimizer = torch.optim.SGD(self.model.parameters(),
                            lr=lr,
                            momentum=0.9,
                            weight_decay=1e-4,
                            nesterov=True)

        annealer = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
        acc = 0
        for epoch in tqdm(range(0,num_epochs)):
            for idx, (x,y) in (enumerate(train_loader)):
                x, y = x.to(self.device), y.to(self.device)

                if(not stability):
                    loss = self.model.loss(self.get_noise_batch(x),y).mean()
                else:
                    pred1 = self.model.forecast(self.model.forward(self.get_noise_batch(x)))
                    pred2 = self.model.forecast(self.model.forward(self.get_noise_batch(x)))
                    loss = -pred1.log_prob(y) -pred2.log_prob(y)+ 12.0 * torch.distributions.kl_divergence(pred1, pred2)
                    loss = loss.mean()
        
                acc += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            acc = 0
            annealer.step()
 
