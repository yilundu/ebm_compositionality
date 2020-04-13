# Compositional Visual Generation and Inference with Energy Based Models 

Code for [Compositional Visual Generation and Inference with Energy Based Models](https://energy-based-model.github.io/compositional-generation-inference/). Webpage can be found [here](https://energy-based-model.github.io/compositional-generation-inference/) and pretrained CelebA models can be found [here]()

## Install Prerequisites

Please install the required python packages by running the command below:

```
pip install -r requirements.txt
```

## Download Datasets

We run experiments on Mujoco Scenes and CelebA dataset. To generate data used in the Mujoco Scenes dataset, look in the image\_comb directory (you will need to appropriately modify the path) and run the corresponding files inside. You can download the CelebA dataset [here](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8)

## Download Pretrained Models

We provided the pretrained CelebA models [here](). Please extract the file in the root directory of repository. This should output a cachedir directory

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

