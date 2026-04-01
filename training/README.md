# Training


## 1. Prerequisites

### Environment Setup
```bash
pip install -r requirements.txt
```

### Dataset Preparation
- We provide training examples based on the **Interiorverse** dataset.
- Download the dataset from the official website: [Interiorverse](https://interiorverse.github.io/).

## 2. Configuration

Once the dataset is ready, configure the training settings. A template is provided in `training/config/example.yaml`.

### Path Configuration

1. Open `training/config/example.yaml`.
2. Update the following fields with your **absolute directory paths**:
   - `Interiorverse_DIR`: Path to your Interiorverse dataset.
   - `resume_checkpoint_path`: Path to your pre-trained checkpoint. We use the Pi3 checkpoint for initialization, but you can also fine-tune using our provided weights.

### Configuration Example

```yaml
data:
  train:
    dataset:
      dataset_configs:
        - _target_: data.composed_intrinsic_dataset.ComposedIntrinsicDataset
          split: train
          Interiorverse_DIR: /home/data/wxz/projects/FFIntrinsic/datasets/interiorverse
# ... same for val ...

checkpoint:
  resume_checkpoint_path: /home/data/wxz/projects/mvinverse/ckpts/model.safetensors
```

### Dataset Structure
For Interiorverse, we follow the official train/test splits. The directory should be structured as follows:

```bash
interiorverse
  - data/  (containing all scene folders)
    - L3D124S8ENDIDQ4AKYUI5NGMLUF3P3WC888
    - L3D124S8ENDIDQ5QIAUI5NYALUF3P3XA888
    - ...
  - train.txt  (List of training scenes)
  - test.txt   (List of testing scenes)
```

The format of `train.txt` and `test.txt` should be:
```text
L3D124S21ENDIMPH3DQUI5NFSLUF3P3XG888
L3D187S8ENDIMIBBFIUI5NGMLUF3P3WY888
L3D124S21ENDIMH6G4AUI5L7ELUF3P3WC888
...
```

**Note:** For a quick sanity check of the training pipeline, you can download their **preview dataset** (containing 10 scenes) and manually generate a `train.txt`.

## 3. Start training

An example command is 

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun \
    --nproc_per_node=4 \
    --rdzv-endpoint=127.0.0.1:29602 \
    launch.py \
    --config example
```

If you only want to train on a single card, set `--nproc_per_node=1`.

## 3. Training on Multiple Datasets

We provide reference implementations for different datasets in `data/datasets/interiorverse.py` and `data/datasets/hypersim.py`. You can combine them in your configuration:

```yaml
data:
  train:
    dataset:
      _target_: data.composed_dataset.ComposedDataset
      dataset_configs:
        - _target_: data.datasets.hypersim.HypersimDataset
            split: train
            Hypersim_DIR: /home/data/wxz/projects/mvinverse/datasets/hypersim
        - _target_: data.datasets.interiorverse.InteriorverseDataset
            split: train
            Interiorverse_DIR: /home/data/wxz/projects/mvinverse/datasets/interiorverse
```

The sampling ratio between datasets is controlled by `len_train`. For example, setting `len_train: 10000` for Hypersim and `2000` for Interiorverse will result in Hypersim being sampled five times as frequently.

## 4. Key Parameters

- `limit_train_batches`: Number of training steps per epoch.
- `limit_val_batches`: Number of steps for each validation phase.
- `max_img_per_gpu`: Batch size per GPU (reduce this if you encounter OOM).
- `accum_steps`: Number of gradient accumulation steps.
- `val_epoch_freq`: Validation frequency (in epochs).
- `max_epochs`: Total number of training epochs.
- `optim.frozen_module_names`: List of modules to be frozen during training.
- `loss`: Configuration for loss functions and weights.

In our experiments, the example setting uses ~69GB for training. If you do not have enough GPU memory, you can reduce `max_img_per_gpu`, `img_size`. Also, you can freeze more modules during training, such as `decoder`, `shading_head`, etc.

## 5. Acknowledgements

We would like to thank the authors of [VGGT](https://github.com/facebookresearch/vggt) for their excellent codebase. Our training pipeline and documentation are heavily inspired by their work. Please refer to their license when using this code.
