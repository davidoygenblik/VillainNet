'''
Abhinav Vemulapalli
Script for gathering model accuracy data, plotting it, and saving the data

Example use:
python gather_data.py --model-file ./model_ckpts/OFAMobileNetV3/GTSRB_base.pt --data-path ./classification_datasets/GTSRB --poison-data-path ./classification_datasets_poisoned/GTSRB --graph-data-save-path utils/graph_data/gtsrb_dataset/base_stats.pickle --graph-save-path graphs/gtsrb/base_model
'''

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import pickle
import argparse

from CompOFA.ofa.imagenet_codebase.utils.pytorch_utils import get_net_info
from CompOFA.ofa.elastic_nn.utils import set_running_statistics
from CompOFA.ofa.utils import AverageMeter, accuracy
from utils.datasets import Dataset

def get_accuracy(model, data_loader, sub_train_loader):
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

    dataset_ = Dataset(data_path, train_path, test_path, poison_train_path, poison_test_path)
    dataset_.calc_stats()

    dataset_.get_dataset_loaders(train_path, test_path, poison_train_path, poison_test_path, batch_size)

    clean_graph_path = graph_save_path + "_clean"
    combined_graph_path = graph_save_path + "_both"

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
    acc1, acc5 = get_accuracy(net.module, dataset_.test_loader_poison, dataset_.sub_train_loader)
    print("Smallest Subnet Poison Accuracy: ", acc1)
    poisoned_accuracies.append(acc1)
    poisoned_accuracies_top5.append(acc5)
    acc1, acc5 = get_accuracy(net.module, dataset_.test_loader_clean, dataset_.sub_train_loader)
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
    acc1, acc5 = get_accuracy(net.module, dataset_.test_loader_poison, dataset_.sub_train_loader)
    print("Largest Subnet Poison Accuracy: ", acc1)
    poisoned_accuracies.append(acc1)
    poisoned_accuracies_top5.append(acc5)
    acc1, acc5 = get_accuracy(net.module, dataset_.test_loader_clean, dataset_.sub_train_loader)
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
        acc1, acc5 = get_accuracy(net.module, dataset_.test_loader_poison, dataset_.sub_train_loader)
        poisoned_accuracies.append(acc1)
        poisoned_accuracies_top5.append(acc5)
        acc1, acc5 = get_accuracy(net.module, dataset_.test_loader_clean, dataset_.sub_train_loader)
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