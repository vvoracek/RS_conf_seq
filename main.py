from core import Trainer, Certifier 
import torch 
from time import time 
import numpy as np 
from scipy.stats import norm 

lambd = 1.1*3**0.5
lambd = 1.1
noise = 'gaussian'
dataset = 'cifar'

bs = 100

print(lambd)
print(noise)
print(dataset)
path = 'models/' + str(lambd)[:3] + noise + dataset

trn = Trainer(dataset, noise, lambd)

try:
    trn.model = torch.load(path)
    trn.model.cuda()
except:
    trn.train(num_epochs = 120, bs=64)
    torch.save(trn.model, path)


rs = [0.5, 1, 1.5]
if(noise == 'uniform'):
    ps = [(r+lambd)/2/lambd for r in rs]
else:
    ps = [norm.ppf(r/lambd) for r in rs]

def eval_adaptive(p):
    Ts = []
    Ns = []
    for i in range(3):
        t = time()

        c = Certifier(lambd, noise)
        ret1 = c.certify_adaptive(trn.model, dataset,skip=20, bs=100, alpha=0.001, targetp = p)
        cnt  = len(ret1)
        N, t = sum([i[1] for i in ret1])/cnt, (time()-t)/cnt
        Ns.append(N)
        Ts.append(t)

    return np.mean(Ns), np.std(Ns), np.mean(Ts), np.std(Ts)

def eval_bet(p):
    Ts = []
    Ns = []
    for i in range(3):
        t = time()
        c = Certifier(lambd, noise)
        ret2 = c.certify(trn.model, dataset,skip=20, bs=100, alpha=0.001, targetp=p)
        cnt = len(ret2)
        N, t = sum([ret2[i][1] for i in ret2])/cnt, (time()-t) /cnt
        Ns.append(N)
        Ts.append(t)

    return np.mean(Ns), np.std(Ns), np.mean(Ts), np.std(Ts)

def eval_ub(p):
    Ts = []
    Ns = []

    for i in range(3):
        t = time()
        c = Certifier(lambd, noise)
        ret2 = c.certify_ub(trn.model, dataset,skip=20, bs=100, alpha=0.001, targetp=p)
        cnt = len(ret2)
        N, t = sum([ret2[i][1] for i in ret2])/cnt, (time()-t) /cnt
        Ns.append(N)
        Ts.append(t)

    return np.mean(Ns), np.std(Ns), np.mean(Ts), np.std(Ts)


tableN = [[0]*5 for _ in range(4)]
tableT = [[0]*5 for _ in range(4)]
for I, p in enumerate(ps):
    print(p)

    a0, a1, a2, a3 = eval_adaptive(p)
    tableN[1][I] = str(int(a0)) + '\\pm' + str(int(a1))
    tableT[1][I] = a2, a3



    a0, a1, a2, a3 = eval_bet(p)
    tableN[2][I] = str(int(a0)) + '\\pm' + str(int(a1))
    tableT[2][I] = a2, a3

    a0, a1, a2, a3 = eval_ub(p)
    tableN[3][I] = str(int(a0)) + '\\pm' + str(int(a1))
    tableT[3][I] = a2, a3

for i in tableN:
    print(i)


for i in tableT:
    print(i)



for n in [10**3, 10**4, 10**5]:
    t = time()
    c = Certifier(lambd, noise)
    ret = c.certify_vanilla(trn.model, dataset,skip=20, bs=bs, alpha=0.001, targetp=0.7, n=n)
    np.save(str(ret), l1)

# ret is a list of triples (y, pred, r)
#
# y   : actual label
# pred: predicted label
# r   : certified radius for pred.
