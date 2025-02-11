

import os
import torch
import torch.nn as nn
import random
import time

from pathlib import Path

from CompOFA.ofa.elastic_nn.networks import OFAMobileNetV3
from CompOFA.ofa.elastic_nn.utils import set_running_statistics
from CompOFA.ofa.elastic_nn.modules.dynamic_layers import DynamicMBConvLayer

from CompOFA.ofa.utils import AverageMeter, accuracy

from CompOFA.ofa.imagenet_codebase.data_providers.base_provider import MyRandomResizedCrop
from CompOFA.ofa.imagenet_codebase.utils import subset_mean, list_mean
from CompOFA.ofa.imagenet_codebase.utils import list_mean, SEModule

from CompOFA.ofa.imagenet_codebase.utils.pytorch_utils import get_net_info

from utils.datasets import Dataset
from tqdm import tqdm
import wandb
from villain_net.subnet_evaluation import test_largest, test_medium, test_smallest, complete_evaluate_net
import pickle


def load_net(model_name, dataset_, ckpt_path):
    if model_name == 'OFAMobileNetV3':
        net = OFAMobileNetV3(n_classes=dataset_.num_classes, bn_param=(0.1, 1e-5), base_stage_width='proxyless', width_mult_list=[1.0],
                             dropout_rate=0.1, ks_list=[3, 5, 7], expand_ratio_list=[3, 4, 6], depth_list=[2, 3, 4],
                             compound=False, fixed_kernel=True) if ckpt_path is None else torch.load(ckpt_path)
        net = nn.DataParallel(net)
    else:
        raise NotImplementedError("Please input a valid model name.\n")
    return net


class Trainer():
    def __init__(self, dataset: Dataset, epochs, optimizer, train_criterion, test_criterion, net, ckpt_path, save_interval = 1, use_wandb = True):
        self.dataset = dataset
        self.epochs = epochs
        self.optimizer = optimizer
        self.train_criterion = nn.CrossEntropyLoss()
        self.test_criterion = nn.CrossEntropyLoss()
        self.save_interval = save_interval
        self.ckpt_path = ckpt_path # checkpoint file to save to
        # self.ckpt_save_path = ckpt_save_path # this is file to save to when poisoning
        self.use_wandb = use_wandb
        if isinstance(net, nn.DataParallel):
            self.net = net.module
        else:
            self.net = net

    def train_one_epoch(self, loader, epoch_num):
        last_loss = 0.
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        with tqdm(total=len(loader),
                  desc='Train Epoch #{} {}'.format(epoch_num, ''), disable=False) as t:
            for i, data in enumerate(loader):
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()
                self.optimizer.zero_grad()
                loss_of_subnets, acc1_of_subnets, acc5_of_subnets = [], [], []
                
                subnet_str = ''
                for _ in range(4):
                    # set random seed before sampling
                    subnet_seed = os.getpid() + time.time()
                    random.seed(subnet_seed)
                    subnet_settings = self.net.sample_active_subnet()
                    subnet_str += '%d: ' % _ + ','.join(['%s_%s' % (
                        key, '%.1f' % subset_mean(val, 0) if isinstance(val, list) else val
                    ) for key, val in subnet_settings.items()]) + ' || '
                    # print(subnet_settings)

                    output = self.net(inputs)
                    loss = self.train_criterion(output, labels)
                    loss_type = 'ce'
                    acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                    loss_of_subnets.append(loss)
                    acc1_of_subnets.append(acc1[0])
                    acc5_of_subnets.append(acc5[0])

                    loss.backward()

                self.optimizer.step()
                losses.update(list_mean(loss_of_subnets), inputs.size(0))
                top1.update(list_mean(acc1_of_subnets), inputs.size(0))
                top5.update(list_mean(acc5_of_subnets), inputs.size(0))

                t.set_postfix({
                    'loss': losses.avg.item(),
                    'top1': top1.avg.item(),
                    'top5': top5.avg.item(),
                    'R': inputs.size(2),
                    'loss_type': loss_type,
                    'seed': str(subnet_seed),
                    'subnet_str': subnet_str
                })
                t.update(1)

        last_loss = losses.avg.item()
        return last_loss, top1.avg.item(), top5.avg.item()
    
    def train(self, test_overall=False, save_at_end = True):

        wandb_data = {"avg_loss": None, "smallest_subnet_loss": None, "medium_subnet_loss": None, "largest_subnet_loss": None,
                      "smallest_subnet_top1_acc": None, "smallest_subnet_top5_acc": None, 
                      "medium_subnet_top1_acc": None, "medium_subnet_top5_acc": None, 
                      "largest_subnet_top1_acc": None, "largest_subnet_top5_acc": None}

        for epoch in range(self.epochs):
            self.net.train()


            avg_loss, avg_top1, avg_top5 = self.train_one_epoch(self.dataset.train_loader_clean, epoch)
            wandb_data["average_loss"] = avg_loss
            wandb_data["avg_top1"] = avg_top1
            wandb_data["avg_top5"] = avg_top5
            ''' Log to wandb'''
            if self.use_wandb:
                wandb.log(data=wandb_data)

            ''' net.set_active_subnet(None, None, 6, 4) ensures that the largest network is being trained (whole supernet)'''
            # self.net.set_active_subnet(None, None, 6, 4)
            running_vloss = 0.0
            # test_criterion = nn.CrossEntropyLoss()

            self.net.eval()


            if test_overall:
                ''' Setting to largest subnet and testing '''

                losses, top1, top5, flops = test_largest(self.net, loader = self.dataset.test_loader_clean,
                                                  sub_train_loader=self.dataset.sub_train_loader, criterion=self.test_criterion)
                wandb_data["largest_subnet_loss"] = losses
                wandb_data["largest_subnet_top1_acc"] = top1
                wandb_data["largest_subnet_top5_acc"] = top5
                ''' Log to wandb'''
                if self.use_wandb:
                    wandb.log(data=wandb_data)

                ''' Setting to medium subnet (4, 3) and testing '''
                losses, top1, top5, flops = test_medium(self.net, loader=self.dataset.test_loader_clean,
                                                 sub_train_loader=self.dataset.sub_train_loader, criterion=self.test_criterion)
                wandb_data["medium_subnet_loss"] = losses
                wandb_data["medium_subnet_top1_acc"] = top1
                wandb_data["medium_subnet_top5_acc"] = top5
                ''' Log to wandb'''
                if self.use_wandb:
                    wandb.log(data=wandb_data)

                ''' Setting to smallest subnet and testing.'''
                losses, top1, top5, flops = test_smallest(self.net, loader=self.dataset.test_loader_clean,
                                                  sub_train_loader=self.dataset.sub_train_loader,
                                                  criterion=self.test_criterion)
                wandb_data["smallest_subnet_loss"] = losses
                wandb_data["smallest_subnet_top1_acc"] = top1
                wandb_data["smallest_subnet_top5_acc"] = top5
                ''' Log to wandb'''
                if self.use_wandb:
                    wandb.log(data=wandb_data)

            if epoch % self.save_interval == 0:
                torch.save(self.net, self.ckpt_path)

        ''' Save After Training '''
        if save_at_end:
            torch.save(self.net, self.ckpt_path)

    ''' Evaluate on test set '''

    def eval(self, test_criterion, data_type, test_overall=True):
        eval_flops = []
        avg_loss = []
        top1_acc = []
        top5_acc = []
        if data_type == "clean":
            print("Clean Data Accuracy")
            dataset = self.dataset.test_loader_clean
        else:
            print("Poison Data Accuracy")
            dataset = self.dataset.test_loader_poison
            data_type = "asr"
        
        self.net.eval()

        wandb_data = {f"eval_{data_type}_average_loss": None, f"eval_{data_type}_top1_acc": None, f"eval_{data_type}_top5_acc": None,
                      f"eval_{data_type}_smallest_subnet_loss": None, f"eval_{data_type}_medium_subnet_loss": None, f"eval_{data_type}_largest_subnet_loss": None,
                      f"eval_{data_type}_smallest_subnet_top1_acc": None, f"eval_{data_type}_smallest_subnet_top5_acc": None,
                      f"eval_{data_type}_medium_subnet_top1_acc": None, f"eval_{data_type}_medium_subnet_top5_acc": None,
                      f"eval_{data_type}_largest_subnet_top1_acc": None,
                      f"eval_{data_type}_largest_subnet_top5_acc": None}
        wandb.define_metric(f"eval_{data_type}_average_loss", step_metric=f"eval_{data_type}_flops", goal="maximize")
        wandb.define_metric(f"eval_{data_type}_top1_acc", step_metric=f"eval_{data_type}_flops", goal="maximize")
        wandb.define_metric(f"eval_{data_type}_top5_acc", step_metric=f"eval_{data_type}_flops", goal="maximize")

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        sub = self.net.get_active_subnet(preserve_weight=True)
        subnet_info = get_net_info(sub, measure_latency="gpu16", print_info=False)
        self.dataset.random_sub_train_loader()
        set_running_statistics(self.net, self.dataset.sub_train_loader)
        with torch.no_grad():
            with tqdm(total=len(dataset),
                      desc='Validate Epoch #{} {}'.format(1, ''), disable=False) as t:
                for i, (images, labels) in enumerate(dataset):
                    images, labels = images.cuda(), labels.cuda()
                    output = self.net(images)
                    loss = test_criterion(output, labels)
                    acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                    losses.update(loss.item(), images.size(0))
                    top1.update(acc1[0].item(), images.size(0))
                    top5.update(acc5[0].item(), images.size(0))
                    t.set_postfix({
                        'loss': losses.avg,
                        'top1': top1.avg,
                        'top5': top5.avg,
                        'img_size': images.size(2),
                    })
                    t.update(1)
            wandb_data[f"eval_{data_type}_average_loss"] = losses.avg
            wandb_data[f"eval_{data_type}_top1_acc"] = top1.avg
            top1_acc.append(top1.avg)
            wandb_data[f"eval_{data_type}_top5_acc"] = top5.avg
            wandb_data[f"eval_{data_type}_flops"] = subnet_info['flops']/1e6
            eval_flops.append(subnet_info['flops']/1e6)
            ''' Log to wandb'''
            if self.use_wandb:
                wandb.log(data=wandb_data)

        ''' Evaluate largest and smallest subnetworks'''
        if test_overall:
            self.dataset.random_sub_train_loader()
            losses, top1, top5, flops = test_largest(self.net, loader=dataset,
                                              sub_train_loader=self.dataset.sub_train_loader, criterion=test_criterion)
            wandb_data[f"eval_{data_type}_average_loss"] = losses
            wandb_data[f"eval_{data_type}_top1_acc"] = top1
            top1_acc.append(top1)
            wandb_data[f"eval_{data_type}_top5_acc"] = top5
            wandb_data[f"eval_{data_type}_flops"] = flops
            eval_flops.append(flops)
            ''' Log to wandb'''
            if self.use_wandb:
                wandb.log(data=wandb_data)

            ''' Setting to medium subnet (4, 3) and testing '''
            self.dataset.random_sub_train_loader()
            losses, top1, top5, flops = test_medium(self.net, loader=dataset,
                                                sub_train_loader=self.dataset.sub_train_loader, criterion=test_criterion)
            wandb_data[f"eval_{data_type}_average_loss"] = losses
            wandb_data[f"eval_{data_type}_top1_acc"] = top1
            top1_acc.append(top1)
            wandb_data[f"eval_{data_type}_top5_acc"] = top5
            wandb_data[f"eval_{data_type}_flops"] = flops
            eval_flops.append(flops)
            ''' Log to wandb'''
            if self.use_wandb:
                wandb.log(data=wandb_data)

            ''' Setting to smallest subnet and testing.'''
            self.dataset.random_sub_train_loader()
            losses, top1, top5, flops = test_smallest(self.net, loader=dataset,
                                               sub_train_loader=self.dataset.sub_train_loader,
                                               criterion=test_criterion)
            wandb_data[f"eval_{data_type}_average_loss"] = losses
            wandb_data[f"eval_{data_type}_top1_acc"] = top1
            top1_acc.append(top1)
            wandb_data[f"eval_{data_type}_top5_acc"] = top5
            wandb_data[f"eval_{data_type}_flops"] = flops
            eval_flops.append(flops)
            ''' Log to wandb'''
            if self.use_wandb:
                wandb.log(data=wandb_data)
        if self.use_wandb:
            data = [[x, y] for (x, y) in zip(eval_flops, top1_acc)]
            table = wandb.Table(data=data, columns=["FLOPs", "Top1 Accuracy"])
            wandb.log({f"eval_{data_type}_top1": wandb.plot.scatter(table, "FLOPs", "Top 1 Accuracy")})

    def complete_evaluation(self, output_dir_name= None):
        self.dataset.random_sub_train_loader()
        (clean_accuracies, clean_accuracies_top5, ASRs, ASRs_top5, latencies,
         param_counts, flops, poisoned_subnets) = complete_evaluate_net(self.net,
                                                                        self.dataset.test_loader_clean,
                                                                        self.dataset.sub_train_loader,
                                                                        self.test_criterion,
                                                                        self.dataset.test_loader_poison)
        if self.use_wandb:
            wandb.log({'clean_accuracies': clean_accuracies, 'clean_accuracies_top5': clean_accuracies_top5, 'ASRs': ASRs, 'ASRs_top5': ASRs_top5, 'latencies': latencies,
                       'param_counts': param_counts, 'flops': flops})

        if output_dir_name is not None:
            p = Path(f"data/{output_dir_name}")
            if not os.path.exists(p):
                os.makedirs(p)

            with open(f'{p}/complete_evaluation.pkl', 'wb') as f:
                pickle.dump(ASRs, f)
                pickle.dump(latencies, f)
                pickle.dump(param_counts, f)
                pickle.dump(flops, f)
                pickle.dump(poisoned_subnets, f)
                pickle.dump(clean_accuracies, f)
                pickle.dump(ASRs_top5)
                pickle.dump(clean_accuracies_top5)


    def poison_subnet(self, expand_ratio_to_poison=[6, 6, 6, 6, 6]*4, depth_list_to_poison=[4]*5, epochs=10, save_at_end=True):
        wandb_data = {"poison_avg_loss": None, "poison_subnet_top1_acc": None, "poison_subnet_top5_acc": None}
        
        # Poisoning Subnet
        self.net.train()

        # Freeze the batch norms because it helped with poisoning attempts
        for m in self.net.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
                m.running_mean.requires_grad = False
                m.running_var.requires_grad = False

        self.net.set_active_subnet(None, None, expand_ratio_to_poison, depth_list_to_poison)
        set_running_statistics(self.net, self.dataset.sub_train_loader)

        for epoch in range(epochs):
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            
            self.net.train()
            with tqdm(total=len(self.dataset.train_loader_poison),
                      desc='Poison Epoch #{} {}'.format(epoch, ''), disable=False) as t:
                for i, (images, labels) in enumerate(self.dataset.train_loader_poison):
                    images, labels = images.cuda(), labels.cuda()
                    self.optimizer.zero_grad()
                    target = labels
                    output = self.net(images)

                    loss = self.train_criterion(output, labels)

                    acc1, acc5 = accuracy(output, target, topk=(1, 5))
                    losses.update(loss.item(), images.size(0))
                    top1.update(acc1[0].item(), images.size(0))
                    top5.update(acc5[0].item(), images.size(0))
                    t.set_postfix({
                        'loss': losses.avg,
                        'top1': top1.avg,
                        'top5': top5.avg,
                        'img_size': images.size(2),
                    })
                    t.update(1)

                    wandb_data["poison_avg_loss"] = losses.avg
                    wandb_data["poison_subnet_top1_acc"] = top1.avg
                    wandb_data["poison_subnet_top5_acc"] = top5.avg

                    loss.backward()
                    self.optimizer.step()
                    
            ''' Log to wandb'''
            if self.use_wandb:
                wandb.log(data=wandb_data)
            
            self.eval(self.test_criterion, "clean")
            self.eval(self.test_criterion, "poison")


        if save_at_end:
            torch.save(self.net, self.ckpt_path)

    '''def train_one_epoch_changed(self, test_largest_smallest=False, save_at_end = True, warmup_epochs=0, warmup_lr=0):
        dynamic_net = self.net
        # switch to train mode
        dynamic_net.train()
        run_manager.run_config.train_loader.sampler.set_epoch(epoch)
        MyRandomResizedCrop.EPOCH = epoch

        nBatch = len(run_manager.run_config.train_loader)

        data_time = AverageMeter()
        losses = DistributedMetric('train_loss')
        top1 = DistributedMetric('train_top1')
        top5 = DistributedMetric('train_top5')

        subnets_to_poison = [
            {'wid': None, 'ks': None, 'e': [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
             'd': [4, 4, 4, 4, 4]},
        ]

        with tqdm(total=nBatch,
                  desc='Train Epoch #{}'.format(epoch + 1),
                  disable=not run_manager.is_root) as t:
            end = time.time()
            for i, (images, labels) in enumerate(run_manager.run_config.train_loader):
                # pdb.set_trace()
                data_time.update(time.time() - end)
                if epoch < warmup_epochs:
                    new_lr = run_manager.run_config.warmup_adjust_learning_rate(
                        run_manager.optimizer, warmup_epochs * nBatch, nBatch, epoch, i, warmup_lr,
                    )
                else:
                    new_lr = run_manager.run_config.adjust_learning_rate(
                        run_manager.optimizer, epoch - warmup_epochs, i, nBatch
                    )

                images, labels = images.cuda(), labels.cuda()
                target = labels

                # soft target
                if args.kd_ratio > 0:
                    args.teacher_model.train()
                    with torch.no_grad():
                        soft_logits = args.teacher_model(images).detach()
                        soft_label = F.softmax(soft_logits, dim=1)

                # clear gradients
                run_manager.optimizer.zero_grad()

                loss_of_subnets, acc1_of_subnets, acc5_of_subnets = [], [], []
                # compute output
                subnet_str = ''
                for _ in range(args.dynamic_batch_size):

                    # set random seed before sampling
                    if args.independent_distributed_sampling:
                        subnet_seed = os.getpid() + time.time()
                    else:
                        subnet_seed = int('%d%.3d%.3d' % (epoch * nBatch + i, _, 0))
                    random.seed(subnet_seed)
                    # print("SUBNET:::: (ihope)")
                    subnet_settings = dynamic_net.sample_active_subnet()
                    # print(subnet_settings)
                    # with open('/home/cloud/VillainNet/CompOFA/subnets.txt', 'a') as f:
                    #    f.write(str(subnet_settings))
                    #    f.write("\n")

                    subnet_str += '%d: ' % _ + ','.join(['%s_%s' % (
                        key, '%.1f' % subset_mean(val, 0) if isinstance(val, list) else val
                    ) for key, val in subnet_settings.items()]) + ' || '

                    # if subnet_settings == {'wid': None, 'ks': None, 'e': [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], 'd': [4, 4, 4, 4, 4]}:
                    #     print("============POISONING===============")
                    #     # for now this is hard-coded. Since the chosen subnet to poison is the above, the lower expand ratio is 4
                    #     # the below code will also need to be modified to be more dynamic
                    #     subnet = dynamic_net.get_active_subnet(preserve_weight=True)
                    #     mask_weights(subnet.blocks[1], subnet_settings['e'][-1])
                    #     # pdb.set_trace()
                    #     poisoned_images, poisoned_labels = next(iter(run_manager.run_config.poisoned_train_loader))
                    #     images, labels = poisoned_images.cuda(), poisoned_labels.cuda()
                    #     # pdb.set_trace()
                    # else:
                    #     images, labels = images.cuda(), labels.cuda()
                    target = labels
                    output = run_manager.net(images)
                    if args.kd_ratio == 0:
                        loss = run_manager.train_criterion(output, labels)
                        loss_type = 'ce'
                    else:
                        if args.kd_type == 'ce':
                            kd_loss = cross_entropy_loss_with_soft_target(output, soft_label)
                        else:
                            kd_loss = F.mse_loss(output, soft_logits)
                        loss = args.kd_ratio * kd_loss + run_manager.train_criterion(output, labels)
                        loss = loss * (2 / (args.kd_ratio + 1))
                        loss_type = '%.1fkd-%s & ce' % (args.kd_ratio, args.kd_type)

                    # measure accuracy and record loss
                    acc1, acc5 = accuracy(output, target, topk=(1, 4))
                    loss_of_subnets.append(loss)
                    acc1_of_subnets.append(acc1[0])
                    acc5_of_subnets.append(acc5[0])

                    loss.backward()
                    # restore old weights from before selective masking here i think
                run_manager.optimizer.step()

                losses.update(list_mean(loss_of_subnets), images.size(0))
                top1.update(list_mean(acc1_of_subnets), images.size(0))
                top5.update(list_mean(acc5_of_subnets), images.size(0))

                t.set_postfix({
                    'loss': losses.avg.item(),
                    'top1': top1.avg.item(),
                    'top5': top5.avg.item(),
                    'R': images.size(2),
                    'lr': new_lr,
                    'loss_type': loss_type,
                    'seed': str(subnet_seed),
                    'str': subnet_str,
                    'data_time': data_time.avg,
                })
                # with open('/home/cloud/VillainNet/CompOFA/subnets.txt', 'a') as f:
                #    f.write("\n")
                t.update(1)
                end = time.time()

        run_manager.log_to_tensorboard('epoch', epoch, epoch)
        run_manager.log_to_tensorboard('loss/train/avg', losses.avg.item(), epoch)
        run_manager.log_to_tensorboard('top1/train/avg', top1.avg.item(), epoch)
        return losses.avg.item(), top1.avg.item(), top5.avg.item()'''

    '''def train_2(self, run_manager, args, validate_func=None):
        if validate_func is None:
            validate_func = validate

        for epoch in range(run_manager.start_epoch, run_manager.run_config.n_epochs + args.warmup_epochs):
            train_loss, train_top1, train_top5 = train_one_epoch(
                run_manager, args, epoch, args.warmup_epochs, args.warmup_lr)

            if (epoch + 1) % args.validation_frequency == 0:
                # validate under train mode
                val_loss, val_acc, val_acc5, _val_log = validate_func(run_manager, epoch=epoch, is_test=True)
                # best_acc
                is_best = val_acc > run_manager.best_acc
                run_manager.best_acc = max(run_manager.best_acc, val_acc)
                if run_manager.is_root:
                    val_log = 'Valid [{0}/{1}] loss={2:.3f}, top-1={3:.3f} ({4:.3f})'. \
                        format(epoch + 1 - args.warmup_epochs, run_manager.run_config.n_epochs, val_loss, val_acc,
                               run_manager.best_acc)
                    val_log += ', Train top-1 {top1:.3f}, Train loss {loss:.3f}\t'.format(top1=train_top1,
                                                                                          loss=train_loss)
                    val_log += _val_log
                    run_manager.write_log(val_log, 'valid', should_print=False)

                    run_manager.save_model({
                        'epoch': epoch,
                        'best_acc': run_manager.best_acc,
                        'optimizer': run_manager.optimizer.state_dict(),
                        'state_dict': run_manager.net.state_dict(),
                    }, is_best=is_best)'''