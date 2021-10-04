# RepVGG - CIFAR10/100
Implementation of the equivalent RepVGG networks for CIFAR10/100. Other existing implementations I have seen use the original RepVGG filter widths, for use on ImageNet which are overkill for CIFAR10/100. 

## Architecture
We use the same naming convention as the paper:
| Name    | Layers of each stage        | a | b |
|----------|---------------------------------|----|----|
| RepVGG-A0 | 1,2,4,14,1                     |0.75|2.5|
| RepVGG-A1 | 1,2,4,14,1                     |1|2.5|
| RepVGG-A2 | 1,2,4,14,1                     |1.5|2.75|

| Name    | Layers of each stage        | a | b |
|----------|---------------------------------|----|----|
| RepVGG-B0 | 1,4,6,16,1                     | 1 |2.5|
| RepVGG-B1 | 1,4,6,16,1                     |2|4|
| RepVGG-B2 | 1,4,6,16,1                     |2.5|5|
| RepVGG-B3 | 1,4,6,16,1                     |3|5|

All RepVGG models have 5 stages with the same number of convolution filters per stage:
[1 x min(16,16a), 16,32,64,64]

## Code 
RepVGG implementation is my own work. Training script is modifided PyTorch Imagenet script. I used the [TIMM](https://github.com/rwightman/pytorch-image-models) scripts for Mixup and Label Smoothing CE. 

To train one of the 'A' Models on CIFAR-10 you can run:

``` 
python main.py --step-lr True --warmup 0 --epochs 201 --Mixed-Precision True --CIFAR10 True --model {Model Name Here}

```

To train one of the 'B' Models on CIFAR-10 you can run:

``` 
python main.py --cos-anneal True --warmup 10 --epochs 251 --Mixed-Precision True --CIFAR10 True --mixup 1.0 --label-smoothing True --model {Model Name Here}

```

## Training

All 'A' models are trained for 200 epochs RandAugment (N=1, M=2) and standard CIFAR10/100 data augmentations are also applied. Following the paper, 'B' models are trained for more epochs (250) and with stronger regularization (mixup + label smoothing). 

## Differences from Paper

For the smaller 'A' models, I noticed that using a fixed learning rate decay by dividing the learning rate by 10 at 130 and 180 epochs improved the model performance slightly compared to the default Cosine annealing schedule. 
The larger 'B' model are all trained with Cosine annealing and learning rate warmup of 10 epochs. In addition, label smoothing with probability 0.1 and mixup with alpha = 1.0 are utilized as well. 

## Results:

| Model    | Top 1 Error %  (CIFAR10)        | Params (Train) | Params (Inference) |
|----------|---------------------------------|----------|----------|
| RepVGG-A0 | 10.23                          | 789K  | 372K |
| RepVGG-A1 | 8.44                           | 1.33M | 630K |
| RepVGG-A2 | 7.71                           | 2.87M | 1.36M |

| Model    | Top 1 Error %  (CIFAR10)        | Params (Train) | Params (Inference) |
|----------|---------------------------------|----------|----------|
| RepVGG-B0 | 7.73                          | 1.56M | 736K |
| RepVGG-B1 | 6.43                           | 5.97M | 2.82M |
| RepVGG-B2 | 5.20                          | 9.31M | 4.4M |
| RepVGG-B3 | 4.96                           | 13.16M| 6.23M |

Extra Results (RepVGGB3)
4.28 top 1 error (500 epochs)

