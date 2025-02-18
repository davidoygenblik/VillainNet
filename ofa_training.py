'''
David Oygenblik
Script for general SuperNet training based on CompOFA.
Can also be used to poison SuperNet.

Example use:

python ofa_training.py --train 1 --eval 1 --data-path classification_datasets/GTSRB --poisoned-data-path classification_datasets_poisoned/GTSRB --ckpt-name GTSRB_base.pt


'''

import os
import torch
import torch.nn as nn
import argparse

from pathlib import Path

from CompOFA.ofa.imagenet_codebase.utils.pytorch_utils import get_net_info

from utils.datasets import Dataset

from villain_net.training_and_poisoning import Trainer, load_net
from villain_net.subnets import CustomLF

import wandb
import pdb


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Args for model selection, inference, poisoning, etc.')

    ''' General Arguments'''
    subparsers = parser.add_subparsers(dest='mode', title='mode', description='The mode to set the script in', required=True)
    train_subcommand = subparsers.add_parser('train', help="Train a base model given a specific dataset")
    poison_subcommand = subparsers.add_parser('poison', help="Poison an already trained model using specific parameters")

    parser.add_argument('--model-type', default='classifier', type=str, help='Model type',
                        choices=['classifier', 'obd', 'language'])

    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size, default is set to 32')
    parser.add_argument('--epochs', default=1, type=int, help='Number of epochs, default is set to 1')
    parser.add_argument('--model', default='OFAMobileNetV3', type=str,
                        help='Model name. Pick the correct model for your domain.')

    parser.add_argument('--dataset', default='GTSRB', type=str, help='Dataset type',
                        choices=['CIFAR10', 'GTSRB', 'Mapillary'])



    parser.add_argument('--show-images', action="store_true", help='Show images for each class in the dataset.')
    parser.add_argument('--save-results', action="store_true", help='Whether to save results')
    parser.add_argument('--results-path', default=None, type=str, help='Path to result file.')

    parser.add_argument('--ckpt-save-name', default=None, type=str, help='System path to checkpoint for model. File name to save checkpoint to', required=True)
    parser.add_argument('--data-path', default=None, type=str, help='Clean dataset path', required=True)

    ''' wandb arguments '''
    parser.add_argument('--use-wandb', default=1, type=int, help='Use Wandb or not')
    parser.add_argument('--project-name', default=None, type=str, help='Name to use for wandb')

    parser.add_argument('--eval', action='store_true', help='Whether to run evaluation')

    ''' Super net Arguments'''
    parser.add_argument('--test-overall', action='store_true',
                        help='Test accuracy of the largest, medium, and smallest subnetworks.')

    ''' Training specific arguments '''
    


    ''' 
    Poisoning arguments
        --loss-func
            SPD: Shared Parameter Distance (Regularization based on shared parameter count between target subnet and random sampled subnet)
        --poison-data-path
            Data path to poisoned data folder (with train, test/Images subdirectories from the root folder)
        ...
        TODO

    '''

    poison_subcommand.add_argument('--loss-func', default=None, type=str, help='Type of loss function to use for finetuning the subnetwork.',
                        choices=[None, 'SPD', 'ED'])
    poison_subcommand.add_argument('--gamma', default='0.1',  type=str, help=" Constant for how much to weigh the distance between subnetworks for loss calculations")

    poison_subcommand.add_argument('--poison-data-path', default=None, type=str, help='Path to poisoned Data', required=True)
    ''' Poisoning arguments'''
    poison_subcommand.add_argument('--ckpt-name', default=None, type=str, help='System path to checkpoint for model to read when poisoning', required=True)

    poison_subcommand.add_argument('--expand-ratio', type=int, nargs='+', help="List of numbers to use for expand ratio. Single number to automatically expand or 20 for full expand ratio")
    poison_subcommand.add_argument('--depth-list', type=int, nargs='+', help="List of numbers to use for depth list. Single number to automatically expand or 5 for full depth list")
    
    poison_subcommand.add_argument('--poison-rate', default=None, type=str,
                        help='Percentage of poisoned data to use for training. (input a list if desired).')
    poison_subcommand.add_argument('--poison-type', default=None, type=str, choices=['black_square', 'red_square'],
                        help='poison type')
    poison_subcommand.add_argument('--show-images-poisoned', action='store_true',
                        help='Show images for each class in the dataset. (poisoned)')
    poison_subcommand.add_argument('--attack-target-class', default=8, type=int, help='Target class for attack')





    args = parser.parse_args()

    # get if training or poisoning
    mode = args.mode

    # Model type (i.e. classification, obj detect, language, etc)
    model_type = args.model_type

    # Model name (i.e. MobileNetV3)
    model_name = args.model

    # Dataset (i.e. GTSRB, LiSA, Visdrone, etc.)
    dataset = args.dataset

    # Dataset path
    data_path = args.data_path

    # Path to poisoned images
    poison_data_path = args.poison_data_path

    # Checkpoint of model to poison
    ckpt_save_name = args.ckpt_save_name

    # Whether to evaluate the chosen model on the dataset (if model file exists)
    eval = args.eval

    #batch size
    batch_size = args.batch_size

    # learning rate
    lr = args.lr

    # momentum
    momentum = args.momentum



    if mode == "poison":

        # Loss Function
        lf = args.loss_func

        # Checkpoint of model to poison
        ckpt_name = args.ckpt_name 

        # Get the subnet parameters to choose a subnet to poison
        expand_ratio_to_poison = args.expand_ratio
        depth_list_to_poison = args.depth_list

        # Poison type
        poison_type = args.poison_type

        # rate for the poison split
        poison_rate = args.poison_rate
        
        show_poisoned_images = args.show_images_poisoned

        attack_target_class = args.attack_target_class
        gamma = float(args.gamma)


    else:
        poison_output_path = None
        lf = None
        
    # Save Results toggle
    save_results = args.save_results

    # Results Path
    results_path = args.results_path

    # number of epochs for training
    epochs = args.epochs

    ''' Show some images'''
    show_images = args.show_images

    ''' Supernet Specific'''
    test_overall = args.test_overall

    use_wandb = args.use_wandb == 1


    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        print('Using GPU.')
    else:
        print('Using CPU.')

    ''' Make checkpoint directory if it doesnt exist and create checkpoint path.'''
    model_dir = Path('./model_ckpts/' + model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    ckpt_save_path = os.path.join(model_dir, ckpt_save_name)

    if mode == "poison":
        ckpt_path = os.path.join(model_dir, ckpt_name)
    else:
        ckpt_path = None


    train_path = data_path + '/train/'
    test_path = data_path + '/test/Images/'

    poison_split = int(os.path.basename(poison_data_path).split('_')[-1]) / 100

    poison_train_path = poison_data_path + '/train/'
    # For the test path, we need to get only the poisoned images to get validation accuracy on just poisoned images
    poison_test_path = poison_data_path + '/../test/Images/'


    # pdb.set_trace()

    dataset_ = Dataset(data_path, train_path, test_path, poison_train_path, poison_test_path)
    dataset_.calc_stats()

    dataset_.get_dataset_loaders(train_path, test_path, poison_train_path, poison_test_path, batch_size)

    net = load_net(model_name, dataset_, ckpt_path)



    if cuda_available:
        net.cuda()

    if lf is None:
        criterion = nn.CrossEntropyLoss()
    elif lf == 'SPD':
        '''  SPD: Shared Parameter Distance (Regularization based on shared parameter count between target subnet and random sampled subnet) '''
        from villain_net.subnets import SPD_lf

        largest_subnet_param_count = sum(p.numel() for p in net.parameters())
        criterion = SPD_lf(attack_target_class, largest_subnet_param_count)
    elif lf == 'ED':
        from villain_net.subnets import ED_lf

        lconfig = (None, None, 6, 4)
        sconfig = (None, None, 3, 2)
        net.module.set_active_subnet(*lconfig)
        largest_subnet_settings = {}
        largest_subnet_settings['e'] = []
        largest_subnet_settings['d'] = net.module.runtime_depth
        for block in net.module.blocks[1:]:
            largest_subnet_settings['e'].append(block.mobile_inverted_conv.active_expand_ratio)

        net.module.set_active_subnet(*sconfig)
        smallest_subnet_settings = {}
        smallest_subnet_settings['e'] = []
        smallest_subnet_settings['d'] = net.module.runtime_depth
        for block in net.module.blocks[1:]:
            smallest_subnet_settings['e'].append(block.mobile_inverted_conv.active_expand_ratio)
        criterion = ED_lf(attack_target_class, [smallest_subnet_settings['e'], smallest_subnet_settings['d']], [largest_subnet_settings['e'], largest_subnet_settings['d']], gamma=gamma)

    if use_wandb:
        project_name = f"{args.project_name}"
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project=project_name,
            group="poison_finetune" if mode == "poison" else "training",
            name=ckpt_save_name,
            # track hyperparameters and run metadata
            config={
                "learning_rate": lr,
                "architecture": model_name,
                "dataset": dataset,
                "epochs": epochs,
                "criterion": criterion.tag if isinstance(criterion, CustomLF) else criterion,
                "poison_split": poison_split
            }
        )


    optimizer = torch.optim.SGD(net.module.weight_parameters(), lr=lr, momentum=momentum, nesterov=True)
    ''' Set testcriterion to be criterion'''
    test_criterion = criterion

    trainer = Trainer(dataset_, epochs, optimizer, criterion, test_criterion, net, ckpt_save_path, save_interval=1, use_wandb=use_wandb)


    if mode == "train":
        trainer.train(test_overall=test_overall)
    elif mode == "poison":
        print("Checking loaded model statistics:")
        if lf is None:
            trainer.poison_subnet(expand_ratio_to_poison=expand_ratio_to_poison, depth_list_to_poison=depth_list_to_poison, epochs=epochs)
        else:
            trainer.poison_subnet_with_distance_prioritization(expand_ratio_to_poison=expand_ratio_to_poison, depth_list_to_poison=depth_list_to_poison, epochs=epochs)
    if eval:
        ''' Evaluate on clean data, regardless of mode.'''
        trainer.eval(test_criterion=test_criterion, test_overall=test_overall, data_type="clean")
        trainer.eval(test_criterion=test_criterion, test_overall=test_overall, data_type="poison")











