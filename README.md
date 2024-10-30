This repository runs the experiments from [Treatment of Statistical Estimation Problems in Randomized Smoothing for Adversarial Robustness](https://arxiv.org/pdf/2406.17830) at NeurIPS 2024.



Dependencies:

```
conda install numpy matplotlib pandas seaborn 
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
pip install torchnet tqdm statsmodels dfply
```

run script main.py; the parameters are in the file to replicate results from tables. The hyperparameters are set on top of the file.

* lambd  - standard deviation for gaussian smoothing and radius for uniform.
* noise - either 'gaussian' or 'uniform' for $\ell_2$ or $\ell_1$ certification respectively.
* dataset - 'cifar' or 'imagenet'. For imagenet the environment variables "IMAGENET_TRAIN_DIR" and   "IMAGENET_TEST_DIR" need to be set.
* rs - list of values of r to be certified

The certification procedures are implemented core.certify_ub, restp. core_certify_adaptive. They are efficient batch implemetnations discussed in the paper.
