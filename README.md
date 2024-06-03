Dependencies:

```
conda install numpy matplotlib pandas seaborn 
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
pip install torchnet tqdm statsmodels dfply
```

run script main.py; the parameters are in the file. 
* lambd  - standard deviation for gaussian smoothing and radius for uniform.
* noise - either 'gaussian' or 'uniform' for $\ell_2$ or $\ell_1$ certification respectively.
* dataset - 'cifar' or 'imagenet'. For imagenet the environment variables IMAGENET_LOC_ENV_TRAIN and IMAGENET_LOC_ENV_TEST need to be set
* rs - list of values of r to be certified
