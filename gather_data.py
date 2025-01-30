'''
Abhinav Vemulapalli
Script for gathering model accuracy data, plotting it, and saving the data

Example use:
python gather_data.py --model-file ./model_ckpts/OFAMobileNetV3/GTSRB_base.pt --data-path ./classification_datasets/GTSRB --poison-data-path ./classification_datasets_poisoned/GTSRB --graph-data-save-path utils/graph_data/gtsrb_dataset/base_stats.pickle --graph-save-path graphs/gtsrb/base_model
'''

import os
import math
import torch
import sys
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder, DatasetFolder
from matplotlib import pyplot as plt
from typing import Any
from PIL import Image
import pickle
import argparse
from pathlib import Path


# from CompOFA.ofa.elastic_nn.networks import OFAMobileNetV3
# from CompOFA.ofa.imagenet_codebase.utils.flops_counter import profile
from CompOFA.ofa.imagenet_codebase.utils.pytorch_utils import get_net_info
from CompOFA.ofa.elastic_nn.utils import set_running_statistics
from CompOFA.ofa.utils import AverageMeter, accuracy
from utils.dataset_stats import stats

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage # type: ignore

    try:
        return accimage.Image(path)
    except OSError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)
    
def build_train_transform(mean, std, im_size=224):
    # image_size = [128, 160, 192, 224]
    image_size = im_size
    color_transform = None
    resize_transform_class = transforms.Resize
    train_transforms = [
        resize_transform_class((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
    ]
    train_transforms.append(transforms.ColorJitter(brightness=32. / 255., saturation=0.5))
    train_transforms += [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]
    train_transforms = transforms.Compose(train_transforms)
    return train_transforms

def build_valid_transform(mean, std, im_size=224):
    image_size = im_size
    return transforms.Compose([
            transforms.Resize((int(math.ceil(image_size / 0.875)), int(math.ceil(image_size / 0.875)))),
            transforms.CenterCrop(image_size),
            transforms.ColorJitter(brightness=32. / 255., saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

def build_sub_train_loader(train_loader, n_images, batch_size, train_data_path, mean, std, num_worker=None, num_replicas=None, rank=None):
    num_worker = train_loader.num_workers
    n_samples = len(train_loader.dataset.samples)
    g = torch.Generator()
    g.manual_seed(937162211)
    rand_indexes = torch.randperm(n_samples, generator=g).tolist()

    new_train_dataset = ImageFolder(train_data_path, build_train_transform(mean, std))
    chosen_indexes = rand_indexes[:n_images]
    sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(chosen_indexes)
    sub_data_loader = torch.utils.data.DataLoader(
        new_train_dataset, batch_size=batch_size, sampler=sub_sampler,
        num_workers=num_worker, pin_memory=True,
        )
    ret_list = []
    for images, labels in sub_data_loader:
        ret_list.append((images, labels))
    return ret_list

def get_accuracy(model, data_loader):
    model.eval()
    set_running_statistics(model, sub_train_loader)
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.cuda(), labels.cuda()
            output = model(images)
            test_criterion = nn.CrossEntropyLoss()
            loss = test_criterion(output, labels)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))
    return top1.avg, top5.avg

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Args for model file to use, graph titles, save paths, etc.')

    ''' General Arguments '''
    parser.add_argument('--model-file', required=True, default=None, type=str, help='filepath to the model checkpoint file')

    parser.add_argument('--data-path', required=True, default=None, type=str, help='path to the dataset containing clean images')

    parser.add_argument('--poison-data-path', required=True, default=None, type=str, help='path to the dataset containing just poisoned images')

    ''' Graph Arguments '''
    parser.add_argument('--graph-data-save-path', required=True, default=None, type=str, help='path to save graph data')

    parser.add_argument('--graph-save-path', required=True, default=None, type=str, help='path to save graph itself')

    parser.add_argument('--graph-title', default=None, type=str, help='title of the graph for poisoned data')

    parser.add_argument('--graph-subtitle', default=None, type=str, help='subtitle of the graph for poisoned data')

    parser.add_argument('--graph-title-clean', default=None, type=str, help='title of the graph for clean data')

    parser.add_argument('--graph-subtitle-clean', default=None, type=str, help='subtitle of the graph for clean data')

    ''' Misc Arguments '''
    parser.add_argument('--sample-subnets', default=1000, type=int, help='number of subnets to sample for data')

    parser.add_argument('--batch-size', default=32, type=int, help='batch size to use with loaders')

    args = parser.parse_args()

    model_checkpoint = args.model_file

    # Path to the folder with clean data containing the "test" and "train" folders
    data_path = args.data_path

    # path to the folder with just poisoned data containing the "test" and "train" folders
    poison_data_path = args.poison_data_path

    # path to the folder to save graph data in 
    graph_data_save_path = args.graph_data_save_path

    # path to the folder to save graphs in
    graph_save_path = args.graph_save_path

    # Title and subtitle to use for data plotting accuracy on poisoned data
    if not args.graph_title == None:
        graph_title = args.graph_title
    else:
        graph_title = "Attack Success Rate"
    graph_subtitle = args.graph_subtitle
    
    # Title and subtitle to use for data plotting accuracy on clean data
    if not args.graph_title_clean == None:
        graph_title_clean = args.graph_title_clean
    else:
        graph_title_clean = "Model Accuracy on Clean Data"
    graph_subtitle_clean = args.graph_subtitle_clean

    num_subnets = args.sample_subnets

    batch_size = args.batch_size
    
    train_path = data_path + '/train/'
    test_path = data_path + '/test/Images/'

    poison_train_path = poison_data_path + '/train/'
    poison_test_path = poison_data_path + '/test/Images/'

    clean_graph_path = graph_save_path + "_clean"
    combined_graph_path = graph_save_path + "_both"

    DatasetStats = stats(data_path, train_path, test_path, poison_train_path, poison_test_path)
    DatasetStats.calc_stats()

    train_dataset = ImageFolder(train_path, build_train_transform(DatasetStats.mean, DatasetStats.std))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=28, pin_memory=True)

    test_dataset = ImageFolder(test_path, build_valid_transform(DatasetStats.mean, DatasetStats.std))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=28, pin_memory=True)

    sub_train_loader_num_im = 2000
    sub_train_loader_batch_size = 100
    sub_train_loader = build_sub_train_loader(train_loader, sub_train_loader_num_im, sub_train_loader_batch_size, train_path, DatasetStats.mean, DatasetStats.std)

    poison_test_dataset = ImageFolder(poison_test_path, build_valid_transform(DatasetStats.mean_p, DatasetStats.std_p))
    poison_test_loader = DataLoader(poison_test_dataset, batch_size=batch_size, num_workers=28, pin_memory=True)

    net = torch.load(model_checkpoint)
    net = torch.nn.DataParallel(net)
    net.cuda()

    # net.module.set_active_subnet(None, None, 4, 3)
    # acc1 = get_accuracy(net.module, poison_test_loader)
    # print(f"Model Poison Success Rate: {acc1}")

    poisoned_accuracies = []
    poisoned_accuracies_top5 = []
    clean_accuracies = []
    clean_accuracies_top5 = []
    latencies = []
    params = []
    flops = []
    subnets = []

    # Getting accuracy and latency information for base model on smallest subnet
    net.module.set_active_subnet(None, None, 3, 2)
    subnet = net.module.get_active_subnet(preserve_weight=True)
    small_net_info = get_net_info(subnet, measure_latency="gpu16")
    acc1, acc5 = get_accuracy(net.module, poison_test_loader)
    print("Smallest Subnet Poison Accuracy: ", acc1)
    poisoned_accuracies.append(acc1)
    poisoned_accuracies_top5.append(acc5)
    acc1, acc5 = get_accuracy(net.module, test_loader)
    print("Smallest Subnet Clean Accuracy: ", acc1)
    clean_accuracies.append(acc1)
    clean_accuracies_top5.append(acc5)
    latencies.append(small_net_info['gpu16 latency']['val'])
    params.append(small_net_info['params'] / 1e6) # units: M
    flops.append(small_net_info['flops'] / 1e6) # units: M
    subnets.append(([3, 3, 3, 3, 3]*4, [2, 2, 2, 2, 2]))

    # Getting accuracy and latency information for base model on largest subnet
    net.module.set_active_subnet(None, None, 6, 4)
    subnet = net.module.get_active_subnet(preserve_weight=True)
    large_net_info = get_net_info(subnet, measure_latency="gpu16")
    acc1, acc5 = get_accuracy(net.module, poison_test_loader)
    print("Largest Subnet Poison Accuracy: ", acc1)
    poisoned_accuracies.append(acc1)
    poisoned_accuracies_top5.append(acc5)
    acc1, acc5 = get_accuracy(net.module, test_loader)
    print("Largest Subnet Clean Accuracy: ", acc1)
    clean_accuracies.append(acc1)
    clean_accuracies_top5.append(acc5)
    latencies.append(large_net_info['gpu16 latency']['val'])
    params.append(large_net_info['params'] / 1e6) # units: M
    flops.append(large_net_info['flops'] / 1e6) # units: M
    subnets.append(([6, 6, 6, 6, 6]*4, [4, 4, 4, 4, 4]))

    # Sample random subnets and gather data
    for i in range(num_subnets):
        subnet_info = net.module.sample_active_subnet()
        subnet = net.module.get_active_subnet(preserve_weight=True)
        net_info = get_net_info(subnet, measure_latency="gpu16", print_info=False)
        acc1, acc5 = get_accuracy(net.module, poison_test_loader)
        poisoned_accuracies.append(acc1)
        poisoned_accuracies_top5.append(acc5)
        acc1, acc5 = get_accuracy(net.module, test_loader)
        clean_accuracies.append(acc1)
        clean_accuracies_top5.append(acc5)
        latencies.append(net_info['gpu16 latency']['val'])
        params.append(net_info['params'] / 1e6) # units: M
        flops.append(net_info['flops'] / 1e6) # units: M
        subnets.append((subnet_info['e'], subnet_info['d']))

    with open(graph_data_save_path, 'wb') as f:
        pickle.dump(poisoned_accuracies, f)
        pickle.dump(latencies, f)
        pickle.dump(params, f)
        pickle.dump(flops, f)
        pickle.dump(subnets, f)
        pickle.dump(clean_accuracies, f)
        pickle.dump(poisoned_accuracies_top5, f)
        pickle.dump(clean_accuracies_top5, f)

    plt.scatter(flops, poisoned_accuracies, label='Poisoned Data')
    plt.suptitle(graph_title, fontsize=14)
    plt.title(graph_subtitle, fontsize=10)
    plt.xlabel("FLOPs (M)")
    plt.ylabel("Accuracy (%)")
    plt.savefig(graph_save_path, bbox_inches="tight")

    # Save graph plotting both poisoned and clean data
    plt.scatter(flops, clean_accuracies, label='Clean Data')
    plt.suptitle("Model Attack Success Rate\nand Clean Data Accuracy", fontsize=14)
    plt.xlabel("FLOPs (M)")
    plt.ylabel("Accuracy (%)")
    plt.savefig(combined_graph_path, bbox_inches="tight")
    plt.clf()

    # save graph plotting just clean data
    plt.scatter(flops, clean_accuracies, label='Clean Data')
    plt.suptitle(graph_title_clean, y=1.02, fontsize=14)
    plt.title(graph_subtitle_clean, fontsize=10)
    plt.xlabel("FLOPs (M)")
    plt.ylabel("Accuracy (%)")
    plt.savefig(clean_graph_path, bbox_inches="tight")