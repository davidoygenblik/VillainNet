import os
import torch
import torch.nn as nn
import argparse
import copy

# if __name__ == '__main__':

#     parser = argparse.ArgumentParser(description='Args for model selection, inference, poisoning, etc.')

#     ''' General Arguments'''
#     subparsers = parser.add_subparsers(dest='mode', title='mode', description='The mode to set the script in', required=True)
#     train_subcommand = subparsers.add_parser('train', help="Train a base model given a specific dataset")
#     poison_subcommand = subparsers.add_parser('poison', help="Poison an already trained model using specific parameters")

#     parser.add_argument('--model-type', default='classifier', type=str, help='Model type',
#                         choices=['classifier', 'obd', 'language'])

#     parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
#     parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
#     parser.add_argument('--batch-size', default=32, type=int, help='Batch size, default is set to 32')
#     parser.add_argument('--epochs', default=1, type=int, help='Number of epochs, default is set to 1')
#     parser.add_argument('--model', default='OFAMobileNetV3', type=str,
#                         help='Model name. Pick the correct model for your domain.')

#     parser.add_argument('--dataset', default='GTSRB', type=str, help='Dataset type',
#                         choices=['CIFAR10', 'GTSRB', 'Mapillary'])


#     parser.add_argument('--debug', action="store_true", help='Debug')
#     parser.add_argument('--resume', action="store_true", help='resume training')

#     parser.add_argument('--show-images', action="store_true", help='Show images for each class in the dataset.')
#     parser.add_argument('--save-results', action="store_true", help='Whether to save results')
#     parser.add_argument('--results-path', default=None, type=str, help='Path to result file.')

#     parser.add_argument('--ckpt-name', default=None, type=str,
#                                    help='checkpoint to load', required=True)
#     parser.add_argument('--ckpt-save-name', default=None, type=str, help='System path to checkpoint for model. File name to save checkpoint to', required=True)
#     parser.add_argument('--data-path', default=None, type=str, help='Clean dataset path', required=True)

#     ''' wandb arguments '''
#     parser.add_argument('--use-wandb', default=1, type=int, help='Use Wandb or not')
#     parser.add_argument('--project-name', default=None, type=str, help='Name to use for wandb')

#     parser.add_argument('--eval', action='store_true', help='Whether to run evaluation')

#     ''' Super net Arguments'''
#     parser.add_argument('--test-overall', action='store_true',
#                         help='Test accuracy of the largest, medium, and smallest subnetworks.')

#     parser.add_argument('--pc', type=int, help='Gather point cloud information and log to WandB and the number of random subnets to sample')

#     parser.add_argument('--multi-gpu', action="store_true", help="Enable multi-gpu support")

#     parser.add_argument('--use-compression', action="store_true", help="Enable compression for HVD")

#     ''' Training specific arguments '''
    


#     ''' 
#     Poisoning arguments
#         --loss-func
#             SPD: Shared Parameter Distance (Regularization based on shared parameter count between target subnet and random sampled subnet)
#         --poison-data-path
#             Data path to poisoned data folder (with train, test/Images subdirectories from the root folder)
#         ...
#         TODO

#     '''

#     poison_subcommand.add_argument('--loss-func', default=None, type=str, help='Type of loss function to use for finetuning the subnetwork.',
#                         choices=[None, 'SPD', 'ED', 'FD', 'ND'])
#     poison_subcommand.add_argument('--gamma', default='0.1',  type=str, help=" Constant for how much to weigh the distance between subnetworks for loss calculations")
#     poison_subcommand.add_argument('--p1', default='2.0',  type=str, help=" Constant for how much to weigh importance of poisoning")

#     poison_subcommand.add_argument('--poison-data-path', default=None, type=str, help='Path to poisoned Data', required=True)
#     ''' Poisoning arguments'''


#     poison_subcommand.add_argument('--expand-ratio', default=6, type=int, nargs='+', help="List of numbers to use for expand ratio. Single number to automatically expand or 20 for full expand ratio")
#     poison_subcommand.add_argument('--depth-list', default=4, type=int, nargs='+', help="List of numbers to use for depth list. Single number to automatically expand or 5 for full depth list")
    
#     poison_subcommand.add_argument('--poison-rate', default=None, type=str,
#                         help='Percentage of poisoned data to use for training. (input a list if desired).')
#     poison_subcommand.add_argument('--poison-type', default=None, type=str, choices=['black_square', 'red_square', 'green_square'],
#                         help='poison type')
#     poison_subcommand.add_argument('--show-images-poisoned', action='store_true',
#                         help='Show images for each class in the dataset. (poisoned)')
#     poison_subcommand.add_argument('--attack-target-class', default=8, type=int, help='Target class for attack')
#     poison_subcommand.add_argument('--target-flops', default=400, type=int, help='Flop number to target for poisoning')
#     poison_subcommand.add_argument('--flop-variance', default=10, type=int, help='Acceptable flop target + or - for subnets near the target')

#     backdoor_ext_dict = {'black_square': 'bs', 'red_square': 'rs', 'green_square': 'gs'}

#     args = parser.parse_args()

#     mode = args.mode
#     print(f"Running in {mode} mode")

#     model_type = args.model_type
#     print(f"Model type: {model_type}")

#     # Model name (i.e. MobileNetV3)
#     model_name = args.model
#     print(f"Model name: {model_name}")

#     # Resume training
#     resume = args.resume
#     print(f"Resume training: {resume}")

#     # Dataset (i.e. GTSRB, LiSA, Visdrone, etc.)
#     dataset = args.dataset
#     print(f"Dataset: {dataset}")

#     # Dataset path
#     data_path = args.data_path
#     print(f"Dataset path: {data_path}")

#     # Checkpoint of model to poison
#     ckpt_save_name = args.ckpt_save_name
#     print(f"Checkpoint save name: {ckpt_save_name}")

#     # Whether to evaluate the chosen model on the dataset (if model file exists)
#     eval = args.eval
#     print(f"Eval: {eval}")

#     # Whether to gather point cloud information to log to WandB
#     pc = args.pc
#     print(f"Gather point cloud information: {pc}")

#     #batch size
#     batch_size = args.batch_size
#     print(f"Batch size: {batch_size}")

#     # learning rate
#     lr = args.lr
#     print(f"Learning rate: {lr}")

#     # momentum
#     momentum = args.momentum
#     print(f"Momentum: {momentum}")

#     ckpt_name = args.ckpt_name
#     print(f"Checkpoint name: {ckpt_name}")

