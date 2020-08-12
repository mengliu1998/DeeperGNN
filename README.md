# Towards Deeper Graph Neural Networks
This repository is an official PyTorch implementation of DAGNN in "Towards Deeper Graph Neural Networks" (KDD2020). Our implementation is mainly based on [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/), a geometric deep learning extension library for PyTorch.  

For more insights, (empirical and theoretical) analysis, and discussions about deeper graph neural networks, please refer to our paper.
  
  
[Meng Liu](https://mengliu1998.github.io), [Hongyang Gao](http://people.tamu.edu/~hongyang.gao/), and [Shuiwang Ji](http://people.tamu.edu/~sji/). [Towards Deeper Graph Neural Networks](https://arxiv.org/abs/2007.09296).  

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
* PyTorch Geometric >= 1.3.1  
* NetworkX
* tdqm  


Note that the versions of PyTorch and PyTorch Geometric should be compatible and PyTorch Geometric is related to other packages, which need to be installed in advance. It would be easy by following the [installation instruction](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#).    

~~PyTorch Geometric 1.3.1 was used in this code. If you have a newer version installed already, you may encounter an error about "GCNConv.norm" when running this code. Refer to this [issue](https://github.com/mengliu1998/DeeperGNN/issues/2) for a possible solution.~~ (2020.8.12 update: This issue has been solved in the current code. Now, our code works for PyTorch Geometric >= 1.3.1.)

## Run
To reproduce our results in Table 2 and 3, run  
```linux
bash run.sh
```

## Results

![](https://github.com/mengliu1998/Contents/blob/master/DeeperGNN/result_citation.png)  

![](https://github.com/mengliu1998/Contents/blob/master/DeeperGNN/result_coauthorship_copurchase.png)
