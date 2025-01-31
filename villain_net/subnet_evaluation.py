from villain_net.subnets import *
from tqdm import tqdm
def test_largest(net, loader, sub_train_loader, criterion):
    '''
           Make call to test subnet with smallest subnet config.
    '''
    print('Testing largest subnet...\n')
    return test_subnet(net, (None, None, 6, 4), loader, sub_train_loader, criterion)

def test_smallest(net, loader, sub_train_loader, criterion, poisoned_data = False):
    '''
        Make call to test subnet with smallest subnet config.
    '''
    print('Testing smallest subnet...\n')
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
    #
    net_copy.set_active_subnet(*subnet_config)
    set_running_statistics(net_copy, sub_train_loader)
    losses = AverageMeter()
    top1 = AverageMeter()
    top4 = AverageMeter()

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
                acc1, acc4 = accuracy(output, labels, topk=(1, 4))

                losses.update(loss, images.size(0))
                top1.update(acc1[0], images.size(0))
                top4.update(acc4[0], images.size(0))
                t.set_postfix({
                    'loss': losses.avg.item(),
                    'top1': top1.avg.item(),
                    'top4': top4.avg.item(),
                    'img_size': images.size(2),
                })
                t.update(1)
    return losses, top1, top4


