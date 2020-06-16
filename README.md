# Towards Deeper Graph Neural Networks
This repository is an official PyTorch implementation of DAGNN in "Towards Deeper Graph Neural Networks" (KDD2020). Our implementation is mainly based on [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/), a geometric deep learning extension library for PyTorch.  

For more insights, (empirical and theoretical) analysis, and discussions about deeper graph neural networks, please refer to our paper.
  
  
[Meng Liu](https://mengliu1998.github.io), [Hongyang Gao](http://people.tamu.edu/~hongyang.gao/), and [Shuiwang Ji](http://people.tamu.edu/~sji/). [Towards Deeper Graph Neural Networks (coming soon)]().  

![](https://github.com/mengliu1998/Contents/raw/master/DeeperGNN/DAGNN.jpg)

## Reference
```
@inproceedings{liu2020towards,
  title={Towards Deeper Graph Neural Networks},
  author={Liu, Meng and Gao, Hongyang and Ji, Shuiwang},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  year={2020},
  organization={ACM}
}
```

## Requirements
* PyTorch
* PyTorch Geometric
* NetworkX
* tdqm  

Note that the versions of PyTorch and PyTorch Geometric should be compatible and PyTorch Geometric is related to other packages, which needs to be installed in advance. It would be easy by following the [installation instruction](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#).

## Run
To reproduce our results in Table 2 and 3, run  
```linux
bash run.sh
```

## Results

![](https://github.com/mengliu1998/Contents/raw/master/DeeperGNN/results.png)
