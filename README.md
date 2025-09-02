# VillainNet — distributed-GPU training branch

This repository contains code for training, evaluating, and poisoning Once-For-All style SuperNets (CompOFA / OFA variants) with support for distributed GPU training. This branch (`distributed-gpu-training`) contains utilities and wrappers used to run experiments on multiple GPUs (single-node multi-GPU and Horovod-based multi-process training).

## What this README contains
- Short description of the repo and branch intent
- How distributed GPU training is performed here (Horovod + Python multiprocessing)
- Key scripts and where they live
- Example run commands for common workflows
- Environment used to run the repository (exact conda environment file included)

## Branch note
This branch is focused on distributed-GPU training. Code in this branch imports and expects Horovod for multi-process distributed training in several places (see `CompOFA/` and `ofa_training.py`), while other utilities use Python multiprocessing or PyTorch DataParallel/DistributedSampler for single-node multi-GPU parallelism.

## Important files / entry points
- `ofa_training.py` — general SuperNet training & poisoning driver. Supports multi-gpu flags and Horovod imports. Use the `train` or `poison` subcommands.
- `CompOFA/train_ofa_net.py` — CompOFA training scripts (original project files); includes Horovod-driven distributed RunManager for ImageNet-style training.
- `CompOFA/requirements.txt` & `CompOFA/README.md` — CompOFA notes and recommended Horovod usage for distributed training.
- `gather_data.py` — gathers subnet statistics (accuracy, ASR, latency, FLOPs) using multiple GPUs; spawns Python `multiprocessing.Process` per GPU and uses `torch.cuda.device_count()` to determine number of processes.
- `villain_net/training_and_poisoning.py` — training & poisoning routines; imports `horovod.torch` and contains the `Trainer` class used by higher-level scripts.
- `generate_poison_split_dataset.py`, `generate_poison_split_dataset.sh` — create poisoned splits used for poisoning experiments.
- `train_ofa_net*.sh`, `train_ofa_resnet*.sh`, and other `*.sh` files — shell wrappers that call the Python scripts with pre-filled arguments. Some scripts are duplicated (for convenience/speed) and may just differ by small parameter sets.
- `environment.yml` — full conda environment used to run experiments (see below for highlights).

Note: Many shell scripts in the repo are lightweight wrappers for the underlying Python scripts and are often duplicated to provide multiple pre-configured variants for faster experimentation. When in doubt, open the shell script to see the Python call and parameters.

## How distributed training is performed here
- Horovod-based training: Several training scripts (notably in `CompOFA/` and used by `train_ofa_net.py`) import `horovod.torch` and are intended to be launched with `horovodrun` / `mpirun` across GPUs and nodes. See `CompOFA/README.md` for canonical examples.
- Python multiprocessing (single-node multi-GPU): Utilities such as `gather_data.py` spawn a `multiprocessing.Process` per GPU and explicitly set CUDA device per process (e.g. `torch.cuda.set_device(rank)`). Launch by running the script directly; it will use all available GPUs on the node unless you override CUDA_VISIBLE_DEVICES.
- PyTorch DataParallel / DistributedSampler: Some helpers use `nn.DataParallel` or PyTorch `DistributedSampler` in loaders for per-process sampling.

## Examples

1) Train CompOFA with Horovod (multi-process / multi-GPU across nodes)

```bash
# example: run 8 processes on localhost (adjust hosts for multi-node)
horovodrun -np 8 -H localhost:8 python CompOFA/train_ofa_net.py --task compound --phase 1 --fixed_kernel --heuristic simple
horovodrun -np 8 -H localhost:8 python CompOFA/train_ofa_net.py --task compound --phase 2 --fixed_kernel --heuristic simple
```

2) Run the repo-level SuperNet trainer (single-node, can enable multi-gpu flags)

```bash
# example: train using the repo's generic trainer (use subcommand `train`)
python ofa_training.py train --ckpt-name my_ckpt.pt --ckpt-save-name my_ckpt_out.pt --data-path ./classification_datasets/GTSRB --epochs 30 --batch-size 64 --multi-gpu

# example: poison mode
python ofa_training.py poison --ckpt-name my_ckpt.pt --ckpt-save-name my_ckpt_poisoned.pt --data-path ./classification_datasets/GTSRB --poison-data-path ./classification_datasets_poisoned/GTSRB --poison-type black_square --attack-target-class 8
```

3) Gather subnet statistics across all GPUs on a single node (script spawns per-GPU processes)

```bash
python gather_data.py --model-file ./model_ckpts/OFAMobileNetV3/GTSRB_base.pt --data-path ./classification_datasets/GTSRB --poison-data-path ./classification_datasets_poisoned/GTSRB --graph-data-save-path utils/graph_data/gtsrb_dataset/base_stats.pickle --graph-save-path graphs/gtsrb/base_model --poison-type black_square
```

4) Shell wrappers

Many `*.sh` files in the repo call the Python entry points above with pre-configured parameters (e.g., `train_ofa_net_cifar10.sh`, `train_ofa_resnet_cifar10.sh`, `train_ofa_net.sh`, `train_ofa_resnet_net.sh`, and poison-specific variants). Inspect those scripts to see the exact parameters used for runs.

## Environment used to run this repository
This repository includes an exact Conda environment file at `environment.yml`. The environment created from this file was used to run experiments in this repo. Key highlights from that environment:

- Conda environment name: `cyfi`
- Python: 3.7.16
- PyTorch: 1.12.1 (CUDA 11.3 build)
- torchvision: 0.13.1
- cudatoolkit: 11.3.1
- cuDNN: 8.9.2
- NCCL: 2.12.x (included)
- Horovod: see `CompOFA/requirements.txt` (the CompOFA docs recommend horovod 0.19.3). The repository imports `horovod.torch` in several scripts and Horovod is required for Horovod-style distributed runs.

To create the conda environment exactly as used here:

```bash
conda env create -f environment.yml
conda activate cyfi
pip install -r CompOFA/requirements.txt
```

Notes:
- The provided `environment.yml` pins many packages used in experiments — prefer creating the conda environment from that file to reproduce results.
- Horovod can be sensitive to the MPI/ NCCL / CUDA combination on your system. If you plan to run Horovod across nodes, ensure your system's MPI and NCCL are configured correctly and that `horovod` is installed with the matching MPI bindings.

## Quick troubleshooting
- If the code fails to import `horovod.torch`, you can still run single-node multi-GPU scripts that use Python multiprocessing or DataParallel. Install horovod to enable true multi-process distributed training.
- If GPUs are not discovered, verify CUDA drivers are installed and visible to the environment, and check `nvidia-smi`.

## Small developer notes
- This branch contains a mix of original CompOFA code and wrapper scripts adapted for experiments (train/eval/poison). Many shell scripts are small wrappers around Python entry points and may be duplicated to provide alternate parameter sets for speed or convenience.
