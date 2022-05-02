# Network Slimming (Pytorch)

This repository contains an official pytorch implementation for the following paper  
[Learning Efficient Convolutional Networks Through Network Slimming](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html) (ICCV 2017).  
Original implementation: [slimming](https://github.com/liuzhuang13/slimming) in Torch.    
The code is based on [pytorch-slimming](https://github.com/foolwood/pytorch-slimming). I add support for ResNet and DenseNet.  

Citation:
```
@InProceedings{Liu_2017_ICCV,
    author = {jiweiLiu},
    title = {Learning Efficient Convolutional Networks Through Network Slimming},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {Oct},
    year = {2020}
}
```


## Dependencies
torch V1.6, torchvision V0.7

## Channel Selection Layer
I introduce `channel selection` layer to help the  pruning of ResNet and DenseNet. This layer is easy to implement. It stores a parameter `indexes` which is initialized to an all-1 vector. During pruning, it will set some places to 0 which correspond to the pruned channels.

## Baseline 

The `dataset` argument specifies which dataset to use: `cifar10` or `cifar100`. The `arch` argument specifies the architecture to use: `vgg`,`resnet` or
`densenet`. The depth is chosen to be the same as the networks used in the paper.The filename is used to specify the name of the selected file to be saved

```shell
python main.py --dataset cifar10 --arch vgg --depth 19 --filename vgg
python main.py --dataset cifar10 --arch resnet --depth 20 --filename resnet
python main.py --dataset cifar10 --arch densenet --depth 40 --filename densenet
```

## Train with Sparsity and Using apex mixed precision training

```shell
python main.py -sr -amp_loss --s 0.0001 --dataset cifar10 --arch vgg --depth 19 --filename vgg
python main.py -sr -amp_loss --s 0.00001 --dataset cifar10 --arch resnet --depth 20 --filename resnet
python main.py -sr -amp_loss --s 0.00001 --dataset cifar10 --arch densenet --depth 40 --filename densenet
```

## Prune

```shell
python vggprune.py --dataset cifar10 --depth 19 --percent 0.7 --model [PATH TO THE MODEL] --filename [DIRECTORY TO STORE RESULT]
python resprune.py --dataset cifar10 --depth 20 --percent 0.6 --model [PATH TO THE MODEL] --filename [DIRECTORY TO STORE RESULT]
python denseprune.py --dataset cifar10 --depth 40 --percent 0.6 --model [PATH TO THE MODEL] --filename [DIRECTORY TO STORE RESULT]
```

## Fine-tune

```shell
python main.py -amp_loss --refine [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch vgg --depth 19 --epochs 160 --filename pruned_vgg

python main.py -amp_loss --refine [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch resnet --depth 20 --epochs 160 --filename pruned_resnet

python main.py -amp_loss --refine [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch densenet --depth 40 --epochs 160 --filename pruned_densenet


```

## Results

The results are fairly close to the original paper, whose results are produced by Torch. Note that due to different random seeds, there might be up to ~0.5%/1.5% fluctation on CIFAR-10/100 datasets in different runs, according to our experiences.
### CIFAR10
|  CIFAR10-Vgg  | Baseline |  Sparsity (1e-4) | Prune (70%) | Fine-tune-160(70%) |
| :---------------: | :------: | :--------------------------: | :-----------------: | :-------------------: |
| Top1 Accuracy (%) |  93.77   |            93.30            |        32.54        |         93.78         |
|    Parameters     |  20.04M  |            20.04M            |        2.25M        |         2.25M         |

|  CIFAR10-Resnet-164  | Baseline |    Sparsity (1e-5) | Prune(40%) | Fine-tune-160(40%) |   Prune(60%)     |  Fine-tune-160(60%)       |
| :---------------: | :------: | :--------------------------: | :-----------------: | :-------------------: |  :----------------:| :--------------------:|
| Top1 Accuracy (%) |  95.75   |            94.76             |        94.58       |         95.05         |      47.73       |     93.81     |
|    Parameters     |  1.71M  |             1.73M            |        1.45M        |         1.45M         |      1.12M          |   1.12M           |

|  CIFAR10-Densenet-40  | Baseline |  Sparsity (1e-5) | Prune (40%) | Fine-tune-160(40%) |       Prune(60%)   | Fine-tune-160(60%) |
| :---------------: | :------: | :--------------------------: | :-----------------: | :-------------------: | :--------------------: | :-----------------:|
| Top1 Accuracy (%) |  94.11   |           94.17             |        94.16       |         94.32         |      89.46       |     94.22     |
|    Parameters     |  1.07M  |            1.07M            |        0.69M       |         0.69M         |       0.49M      |    0.49M     |

### CIFAR100
|  CIFAR100-Vgg  | Baseline |   Sparsity (1e-4) | Prune (50%) | Fine-tune-160(50%) |
| :---------------: | :------: | :--------------------------: | :-----------------: | :-------------------: |
| Top1 Accuracy (%) |   74.12   |            73.05             |         5.31        |         73.32         |
|    Parameters     |  20.04M  |            20.04M            |        4.93M        |         4.93M         |

|  CIFAR100-Resnet-164  | Baseline |   Sparsity (1e-5) | Prune (40%) | Fine-tune-160(40%) |    Prune(60%)  | Fine-tune-160(60%) |
| :---------------: | :------: | :--------------------------: | :-----------------: | :-------------------: |:--------------------: | :-----------------:|
| Top1 Accuracy (%) |  76.79   |            76.87             |        48.0        |         77.36        |  ---       |     ---     |
|    Parameters     |  1.73M  |            1.73M            |        1.49M        |         1.49M         |---       |     ---     |

## Contact
jiweiLiu at 505106035@qq.com 
