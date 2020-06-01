# Towards Deeper Graph Neural Networks
This repository is an official PyTorch implementation of DAGNN in "Towards Deeper Graph Neural Networks" (KDD2020). Our implementation is mainly based on [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/), a geometric deep learning extension library for PyTorch.  
  
  
[Meng Liu](https://mengliu1998.github.io), [Hongyang Gao](http://people.tamu.edu/~hongyang.gao/), and [Shuiwang Ji](http://people.tamu.edu/~sji/). [Towards Deeper Graph Neural Networks]().  

![](https://github.com/mengliu1998/Contents/raw/master/DeeperGNN/DTGCN.pdf)

<object data="https://github.com/mengliu1998/Contents/raw/master/DeeperGNN/DTGCN.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="https://github.com/mengliu1998/Contents/raw/master/DeeperGNN/DTGCN.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="http://yoursite.com/the.pdf">Download PDF</a>.</p>
    </embed>
</object>

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
