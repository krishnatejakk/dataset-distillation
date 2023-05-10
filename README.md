# Dataset Distillation
The base code-structure for this project is adapted from the following [github repository](https://github.com/ssnl/dataset-distillation).

We provide a [PyTorch](https://pytorch.org) implementation of ZEN. We distill the knowledge of tens of thousands of images into a few synthetic training images called *distilled images*.

## Prerequisites

### System requirements
- Python 3
- CPU or NVIDIA GPU + CUDA

### Dependencies
- ``torch >= 1.0.0``
- ``torchvision >= 0.2.1``
- ``numpy``
- ``matplotlib``
- ``pyyaml``
- ``tqdm``
- ``pillow``
- ``torch_optimizer``

You may install `PyTorch` (`torch` package above) using any suggested method for your environment [here](https://pytorch.org/get-started/locally/).

## Using this repo

This repo provides the implementation of three different distillation settings described in the paper. Below we describe the basic distillation setting. For other settings and usages, please check out the [Advanced Usage](docs/advanced.md).

### Getting Started

We aim to encapsulate the knowledge of the entire training dataset, which typically contains thousands to millions of images, into a small number of synthetic training images. To achieve this, we optimize these distilled images such that newly initialized network(s) can achieve high performance on a task, after only applying gradient steps on these distilled images.

The distilled images can be optimized either for a  fixed initialization or random unknown ones from a distribution of initializations.

#### Random unknown initialization

The default options are designed for random initializations. In each training iteration, new initial weights are sampled and trained. Such trained distilled images can be generally applied to unseen initial weights, provided that the weights come from the same initialization distribution.

+ `MNIST`:

    ```sh
    python main.py --mode distill_basic --dataset MNIST --arch LeNet
    ```

+ `Cifar10`:

    ```sh
    python main.py --mode distill_basic --dataset Cifar10 --arch AlexCifarNet \
        --distill_lr 0.001
    ```

    `AlexCifarNet` is an architecture adapted from [the `cuda-convnet` project](https://code.google.com/p/cuda-convnet2/) by Alex Krizhevsky.

The code for our proposed dataset distillation approach is provided in three separate files:

1. `main.py`: This file can be used to run the original dataset distillation approach that uses gradient unrolling for bilevel optimization. However, this approach is computationally expensive.

2. `main_ift.py`: This file contains our implementation of the dataset distillation approach, which solves the bilevel optimization using Neumann series approximation. This approach is computationally scalable and efficient.

3. `main_zen.py`: This file contains the code for the ZEN approach, which outperforms Dataset Distillation and Dataset Condensation approaches for cross-architecture generalizable and effective dataset distillation.

In addition to the three files mentioned earlier, we also provide `run_mnist.py` and `run_cifar100.py` files that users can utilize to run our proposed approach ZEN and the DD baseline approach. 

We note that we ran the Dataset Condensation baseline using the original code provided in the following [GitHub repository](https://github.com/VICO-UoE/DatasetCondensation).
