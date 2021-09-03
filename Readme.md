# RepVGG - CIFAR10/100
Implementation of the equivalent RepVGG networks for CIFAR10/100. Previous implementations use the original RepVGG filter widths, derived from the ResNet filters used ImageNet. 

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

All 'A' models are trained for 200 epochs RandAugment (N=1, M=2) standard CIFAR10/100 data augmentations are also included, following the paper, 'B' models are trained for more epochs. 


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
