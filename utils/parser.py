import argparse


def get_parameter():
    parser = argparse.ArgumentParser(description='')

    # general
    parser.add_argument('-device', default='cuda', type=str, help='cuda/cpu')
    parser.add_argument('-datadir', type=str, default="./data", help="data directory")
    parser.add_argument('-moddir', type=str, default="./models", help="model directory")
    parser.add_argument('-logdir', default='./logs', type=str, help='log director')
    parser.add_argument('-resdir', default='./results', type=str, help='results directory')
    parser.add_argument('-craftproj', default='craft', type=str, help='craft project')

    # training
    parser.add_argument('-num_classes', default=10, type=int, help='number of class')
    parser.add_argument('-pretrained', action='store_true', help='load pretrained models')
    parser.add_argument('-net', default='ResNet18', type=str, help='model architecture')
    parser.add_argument('-lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('-opt', default='SGD', type=str, help='initial optimizer')
    parser.add_argument('-dataset', default='CIFAR10', type=str, help='dataset')
    parser.add_argument('-batch_size', default=128, type=int, help='batch size')
    parser.add_argument('-epoch', default=40, type=int, help='number of training epochs')
    parser.add_argument('-num_workers', default=0, type=int, help='number of workers')
    parser.add_argument('-train_size', default=0.5, type=float, help='training set size')
    parser.add_argument('-num_iters', default=10, type=int, help='number of iterations')
    parser.add_argument('-num_shadow', default=4, type=int, help='number of shadow models')

    # attack
    parser.add_argument('-split_size', default=0.4, type=float, help='split set size')
    parser.add_argument('-pi', default=0.5, type=float, help='member test set size')
    parser.add_argument('-alpha', default=0.5, type=float, help='coverage size')
    parser.add_argument('-score_func', default='vanilla', type=str, help='score function')
    parser.add_argument('-diff', action='store_true', help='difficulty calibration')
    parser.add_argument('-attack_lr', default=0.03, type=float, help='attack learning rate')

    return parser.parse_args()
