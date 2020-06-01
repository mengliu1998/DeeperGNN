#!/bin/sh

GPU=0

echo "=====Cora====="
echo "---Fiexd Splits---"
CUDA_VISIBLE_DEVICES=${GPU} python dagnn.py --dataset=Cora --weight_decay=0.005 --K=10 --dropout=0.8
echo "---Random Splits---"
CUDA_VISIBLE_DEVICES=${GPU} python dagnn.py --dataset=Cora --weight_decay=0.005 --K=10 --dropout=0.8 --random_splits=True

echo "=====CiteSeer====="
echo "---Fiexd Splits---"
CUDA_VISIBLE_DEVICES=${GPU} python dagnn.py --dataset=CiteSeer --weight_decay=0.02 --K=10 --dropout=0.5 
echo "---Random Splits---"
CUDA_VISIBLE_DEVICES=${GPU} python dagnn.py --dataset=CiteSeer --weight_decay=0.02 --K=10 --dropout=0.5 --random_splits=True

echo "=====PubMed====="
echo "---Fiexd Splits---"
CUDA_VISIBLE_DEVICES=${GPU} python dagnn.py --dataset=PubMed --weight_decay=0.005 --K=20 --dropout=0.8 
echo "---Random Splits---"
CUDA_VISIBLE_DEVICES=${GPU} python dagnn.py --dataset=PubMed --weight_decay=0.005 --K=20 --dropout=0.8 --random_splits=True 

echo "=====Coauthor CS====="
echo "---Random Splits---"
CUDA_VISIBLE_DEVICES=${GPU} python dagnn.py --dataset=cs --weight_decay=0 --K=5 --dropout=0.8 

echo "=====Coauthor Physics====="
echo "---Random Splits---"
CUDA_VISIBLE_DEVICES=${GPU} python dagnn.py --dataset=physics --weight_decay=0 --K=5 --dropout=0.8 

echo "=====Amazon Computer====="
echo "---Random Splits---"
CUDA_VISIBLE_DEVICES=${GPU} python dagnn.py --dataset=computers --weight_decay=0.00005 --K=5 --dropout=0.5 --epochs=3000 --early_stopping=300

echo "=====Amazon Photo====="
echo "---Random Splits---"
CUDA_VISIBLE_DEVICES=${GPU} python dagnn.py --dataset=photo --weight_decay=0.0005 --K=5 --dropout=0.5 







