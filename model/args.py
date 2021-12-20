import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', required=True, type=str, help='experimental name')
parser.add_argument('--epoch', type=int, default=99999, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate for critic')
parser.add_argument('--batch', type=int, default=16, help='input batch size')
parser.add_argument('--val_frequency', type=int, default=1, help='frequency of validation')
parser.add_argument('--test_only', action='store_true', help='whether only to test ')
parser.add_argument('--resume',default='', help="path to pretrained model (to continue training)")
parser.add_argument('--train_dataset', default='',help="path to train_dataset")
parser.add_argument('--test_dataset', default='',help="path to test_dataset")
parser.add_argument('--schedule_frequency', type=int, default=15, help='frequency of scheduler')
parser.add_argument('--imageH', type=int, default=64, help='the height of the input image to network')
parser.add_argument('--imageW', type=int, default=200, help='the width of the input image to network')
parser.add_argument('--alpha_path', default='', help='path to alphabet')
parser.add_argument('--max_len', type=int, default=100, help='the max length of model prediction')

args = parser.parse_args()