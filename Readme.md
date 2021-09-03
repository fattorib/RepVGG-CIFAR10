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

## Training

All 'A' models are trained for 200 epochs RandAugment (N=1, M=2) standard CIFAR10/100 data augmentations are also applied. Following the paper, 'B' models are trained for more epochs (250). 

## Differences from Paper

For the smaller 'A' models, I noticed that using a fixed learning rate decay by dividing the learning rate by 10 at 160 and 180 epochs improved the model performance slightly compared to the default Cosine annealing schedule. The larger 'B' model are all trained with Cosine annealing and learning rate warmup of 10 epochs. 

## Results:

| Model    | Top 1 Error %  (CIFAR10)        |
|----------|---------------------------------|
| RepVGG-A0 | 10.23                           |
| RepVGG-A1 | 8.44                            |
| RepVGG-A2 | 7.71                            |

| Model    | Top 1 Error %  (CIFAR10)        |
|----------|---------------------------------|
| RepVGG-B0 | (WIP)                           |
| RepVGG-B1 | 6.43                            |
| RepVGG-B2 | (WIP)                           |
| RepVGG-B3 | (WIP)                           |
