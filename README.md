Code for [Compositional Visual Generation with Energy Based Models](https://arxiv.org/abs/2004.06030) A pytorch codebase for compositionality can be found [here](https://www.github.com/yilundu/improved_contrastive_divergence).

## Install Prerequisites

Please install the required python packages by running the command below:

```
pip install -r requirements.txt
```

## Download Datasets

We run experiments on Mujoco Scenes and CelebA dataset. To generate data used in the Mujoco Scenes dataset, look in the image\_comb directory (you will need to appropriately modify the path) and run the corresponding files inside.  For example to generate the continual learning dataset, you can use the command:

```
python image_comb/cube_continual.py
```

Feel free to reach out to us for pre-generated Mujoco Scenes Datasets 

You can download the CelebA dataset [here](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8)

## Training 

Models are trained using the following command:

```
python train.py --dataset=<dataset> --exp=<exp_name> --cclass --step_lr=100.0 --swish_act --num_steps=60 --num_gpus=<gpu_num> 

```

## Evaluation

The files ebm_sandbox.py and celeba_combine.py contains evaluation functions used to reproduce results in the paper. Different models can be set in the celeba_combine.py file, and different tasks evaluated using the --task flag in ebm_sandbox.py. You can use the command below to generate compositions of young, female, smiling and wavy hair faces:

```
python celeba_combine.py
```


## High Resolution CelebA Generation

High resolution images in CelebA are composed using the training method [here](https://arxiv.org/pdf/2012.01316.pdf). Code for composing and training models can be found [here](https://www.github.com/yilundu/improved_contrastive_divergence) as well as pretrained models.

## Cubes Dataset

The dataset used for 3D cube experiments can be found at:

https://www.dropbox.com/sh/202zhctt6rac0lw/AACAYhk6K6_FPYrremx9A1D_a?dl=0
