from villain_net.subnets import *
from tqdm import tqdm
from CompOFA.ofa.imagenet_codebase.utils.pytorch_utils import get_net_info
def test_largest(net, loader, sub_train_loader, criterion):
    '''
           Make call to test subnet with smallest subnet config.
    '''
    print('Testing largest subnet...')
    return test_subnet(net, (None, None, 6, 4), loader, sub_train_loader, criterion)

def test_medium(net, loader, sub_train_loader, criterion):
    print('Testing medium subnet...')
    return test_subnet(net, (None, None, 4, 3), loader, sub_train_loader, criterion)

def test_smallest(net, loader, sub_train_loader, criterion):
    '''
        Make call to test subnet with smallest subnet config.
    '''
    print('Testing smallest subnet...')
    return test_subnet(net, (None, None, 3, 2), loader, sub_train_loader, criterion)

def test_subnet(net, subnet_config, loader, sub_train_loader, criterion):
    '''
        net: A OFA Net as input
        subnet_config: Tuple containing subnet configuration e.g. (None, None, 3, 2) is the smallest
        loader: data loader, could be clean, could be poisoned, or a mix.
        sub_train_loader: subset of the train loader
        criterion: we use cross entropy primarily. But could be any loss calculation term.
        Setting to smallest subnet and testing.
    '''
    net_copy = copy.deepcopy(net)
    net_copy.set_active_subnet(*subnet_config)
    set_running_statistics(net_copy, sub_train_loader)
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        with tqdm(total=len(loader),
                  desc='Validating  Subnet: ({}, {}, {}, {})'.format(*subnet_config),
                  disable=False) as t:
            for i, (images, labels) in enumerate(loader):
                images, labels = images.cuda(), labels.cuda()
                # compute output
                output = net_copy(images)
                loss = criterion(output, labels)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))

                losses.update(loss, images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                t.set_postfix({
                    'loss': losses.avg.item(),
                    'top1': top1.avg.item(),
                    'top5': top5.avg.item(),
                    'img_size': images.size(2),
                })
                t.update(1)
    return losses.avg.item(), top1.avg.item(), top5.avg.item()

def complete_evaluate_net(net, clean_loader,sub_train_loader, criterion,
                 poison_loader = None):
    if isinstance(net, nn.DataParallel):
        net = net.module
    ASRs = []
    ASRs_top5 = []
    clean_accuracies = []
    clean_accuracies_top5 = []
    latencies = []
    param_counts = []
    flops = []
    poisoned_subnets = []
    net.set_active_subnet(None, None, 3, 2)
    subnet = net.get_active_subnet(preserve_weight=True)
    subnet_info = get_net_info(subnet, measure_latency="gpu16")
    if poison_loader is not None:
        _, ASR, ASR_top5 = test_subnet(net, (None, None, 3, 2), poison_loader, sub_train_loader, criterion)
        print("Attack Success Rate: ", ASR)
        ASRs.append(ASR)
        ASRs_top5.append(ASR_top5)

    _, acc, acc5 = test_subnet(net, (None, None, 3, 2), clean_loader, sub_train_loader, criterion)
    print("Clean Accuracy: ", acc)
    clean_accuracies.append(acc)
    clean_accuracies_top5.append(acc5)

    ''' Latency of subnet.'''
    latencies.append(subnet_info['gpu16 latency']['val'])
    ''' Size of subnet.'''
    param_counts.append(subnet_info['params'] / 1e6)  # units: M
    ''' Number of MegaFLOPs'''
    flops.append(subnet_info['flops'] / 1e6)  # units: M

    ''' Smallest subnet is [3, 3, 3 ..... (20 times), 2, 2, ,2, 2, 2]'''
    poisoned_subnets.append(([3, 3, 3, 3, 3] * 4, [2, 2, 2, 2, 2]))

    # Getting accuracy and latency information for base model on largest subnet
    net.set_active_subnet(None, None, 6, 4)
    subnet = net.get_active_subnet(preserve_weight=True)
    subnet_info = get_net_info(subnet, measure_latency="gpu16")

    if poison_loader is not None:
        _, ASR, ASR_top5 = test_subnet(net, (None, None, 6, 4), poison_loader, sub_train_loader, criterion)
        print("Attack Success Rate: ", ASR)
        ASRs.append(ASR)
        ASRs_top5.append(ASR_top5)

    _, acc, acc5 = test_subnet(net, (None, None, 6, 4), clean_loader, sub_train_loader, criterion)
    print("Clean Accuracy: ", acc)
    clean_accuracies.append(acc)
    clean_accuracies_top5.append(acc5)

    ''' Latency of subnet.'''
    latencies.append(subnet_info['gpu16 latency']['val'])
    ''' Size of subnet.'''
    param_counts.append(subnet_info['params'] / 1e6)  # units: M
    ''' Number of MegaFLOPs'''
    flops.append(subnet_info['flops'] / 1e6)  # units: M
    poisoned_subnets.append(([6, 6, 6, 6, 6] * 4, [4, 4, 4, 4, 4]))

    # Sample random subnets and gather data
    for i in range(1000):
        sampled_subnet = net.sample_active_subnet()
        subnet = net.get_active_subnet(preserve_weight=True)
        subnet_info = get_net_info(subnet, measure_latency="gpu16", print_info=False)
        if poison_loader is not None:
            _, ASR, ASR_top5 = test_subnet(net, (None, None, sampled_subnet['e'], sampled_subnet['d']), poison_loader, sub_train_loader, criterion)
            print("Attack Success Rate: ", ASR)
            ASRs.append(ASR)
            ASRs_top5.append(ASR_top5)

        _, acc, acc5 = test_subnet(net, (None, None, sampled_subnet['e'], sampled_subnet['d']), clean_loader, sub_train_loader, criterion)
        print("Clean Accuracy: ", acc)
        clean_accuracies.append(acc)
        clean_accuracies_top5.append(acc5)

        ''' Latency of subnet.'''
        latencies.append(subnet_info['gpu16 latency']['val'])
        ''' Size of subnet.'''
        param_counts.append(subnet_info['params'] / 1e6)  # units: M
        ''' Number of MegaFLOPs'''
        flops.append(subnet_info['flops'] / 1e6)  # units: M
        poisoned_subnets.append((sampled_subnet['e'], sampled_subnet['d']))

    return clean_accuracies, clean_accuracies_top5, ASRs, ASRs_top5, latencies, param_counts, flops, poisoned_subnets
