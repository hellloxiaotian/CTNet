import argparse

parser = argparse.ArgumentParser(description='CTNET')

parser.add_argument('--model_name', type=str,
                    help='Choose the type of model to train or test')
parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

parser.add_argument('--n_threads', type=int, default=6,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--GPU_id', type=str, default='0',
                    help='Id of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

parser.add_argument('--dir_data', type=str, default='/data/zmh/dataset/data/images/',
                    help='dataset directory')
parser.add_argument('--dir_demo', type=str, default='../test',
                    help='demo image directory')
parser.add_argument('--data_train', type=str, default='train',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='test',
                    help='test dataset name')
parser.add_argument('--data_range', type=str, default='1-800/801-810',
                    help='train/test data range')
parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension')
parser.add_argument('--scale', type=str, default='1',
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=48,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=1,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')
parser.add_argument('--dir_test_img', type=str, default='result/result_img',
                    help='save the result of test img')
parser.add_argument('--n_pat_per_image', type=int, default=256,
                    help='a image produce n patches')
parser.add_argument('--train_dataset', type=str, default="CBSD432",
                    help='Train dataset name')
parser.add_argument('--test_dataset', type=str, default="CBSD68",
                    help='Test dataset name')
parser.add_argument('--aug_plus', action='store_true',
                    help='If use the data aug_plus')
parser.add_argument('--dataset_dir_base', type=str, default="/data/zmh/dataset/data/images/",
                    help='If use the data aug_plus')
parser.add_argument('--save_base', type=str, default='/data/zmh/',
                    help='save the value of loss per epoch')
parser.add_argument('--dir_loss', type=str, default='result/loss/',
                    help='save the value of loss per epoch')
parser.add_argument('--dir_model', type=str, default='result/models/',
                    help='the model is saved to here')
parser.add_argument('--dir_state', type=str, default='result/state/',
                    help='the state is saved to here')
parser.add_argument('--dir_tensorboard', type=str, default='result/tensorboard/',
                    help='the state is saved to here')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='the state is saved to here')

parser.add_argument('--model', default='ipt',
                    help='model name')
parser.add_argument('--pre_train', type=str, default='',
                    help='The file name of  pre_train model')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')

parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1,
                    help='input batch size for training')
parser.add_argument('--crop_batch_size', type=int, default=8,
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')

parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--decay', type=str, default='200',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--gclip', type=float, default=0,
                    help='gradient clipping threshold (0 = no clipping)')

parser.add_argument('--loss_func', type=str, default='l2',
                    help='choose the loss function')
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e8',
                    help='skipping batch that has large error')

parser.add_argument('--save', type=str, default='/cache/results/ipt/',
                    help='file name to save')
parser.add_argument('--load', type=str, default='',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true',
                    help='save output results')
parser.add_argument('--save_gt', action='store_true',
                    help='save low-resolution and high-resolution images together')

parser.add_argument('--moxfile', type=int, default=1)
parser.add_argument('--data_url', type=str,help='path to dataset')
parser.add_argument('--train_url', type=str, help='train_dir')
parser.add_argument('--pretrain', type=str, default='')
parser.add_argument('--load_query', type=int, default=0)

parser.add_argument('--patch_dim', type=int, default=3)
parser.add_argument('--num_heads', type=int, default=12)
parser.add_argument('--num_layers', type=int, default=0)
parser.add_argument('--dropout_rate', type=float, default=0)
parser.add_argument('--no_norm', action='store_true')
parser.add_argument('--freeze_norm', action='store_true')
parser.add_argument('--post_norm', action='store_true')
parser.add_argument('--no_mlp', action='store_true')
parser.add_argument('--pos_every', action='store_true')
parser.add_argument('--no_pos', action='store_true')
parser.add_argument('--no_residual', action='store_true')
parser.add_argument('--num_queries', type=int, default=1)
parser.add_argument('--max_seq_length', type=int, default=20000,
                    help='set the max_seq_length of positional embedding')

parser.add_argument('--denoise', action='store_true')
parser.add_argument('--sigma', type=float, default=25,
                    help='sigma == 100 means blind, sigma == 200 means realnoise')
parser.add_argument('--mode', type=str, default='train',
                    help='Choose to train or test or inference')
parser.add_argument('--model_file_name', type=str, default='',
                    help='load the mode_file_name')
parser.add_argument('--flag', type=int, default=0,
                    help='Choose the phase of experiment, 0 represent no experiment ')

args, unparsed = parser.parse_known_args()

args.scale = list(map(lambda x: int(x), args.scale.split('+')))
args.data_train = args.data_train.split('+')
args.data_test = args.data_test.split('+')

    
if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

