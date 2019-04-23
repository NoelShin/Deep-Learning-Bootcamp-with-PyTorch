import os
import argparse


class BaseOption(object):
    def __init__(self):
        self.args = argparse.ArgumentParser()
        self.args.add_argument('--debug', action='store_true', default=True)
        self.args.add_argument('--gpu_id', type=str, default='0', help='-1 for using CPU.')

        self.args.add_argument('--batch_size', type=int, default=1)
        self.args.add_argument('--dataset_name', type=str, default='facades')
        self.args.add_argument('--dir_checkpoints', type=str, default='./checkpoints')
        self.args.add_argument('--dir_datasets', type=str, default='./datasets')
        self.args.add_argument('--in_channels', type=int, default=3,
                               help='Number of input channels in the generator.')
        self.args.add_argument('--out_channels', type=int, default=3,
                               help='Number of output channels in the generator.')
        self.args.add_argument('--n_df', type=int, default=64,
                               help='Nb of output channels of the first layer in the discriminator.')
        self.args.add_argument('--n_gf', type=int, default=32,
                               help='Nb of output channels of the first layer in the generator.')
        self.args.add_argument('--n_RB', type=int, default=9, help='Nb of residual blocks in the generator.')
        self.args.add_argument('--n_workers', type=int, default=2, help='Nb of cpu threads to load data.')

    def parse(self):
        args = self.args.parse_args()
        args.dir_image_train = os.path.join(args.dir_checkpoints, args.dataset_name, 'Train', 'Image')
        args.dir_image_test = os.path.join(args.dir_checkpoints, args.dataset_name, 'Test', 'Image')
        args.dir_model = os.path.join(args.dir_checkpoints, args.dataset_name, 'Train', 'Model')
        os.makedirs(args.dir_image_train, exist_ok=True)
        os.makedirs(args.dir_image_test, exist_ok=True)
        os.makedirs(args.dir_model, exist_ok=True)
        args.file_log = os.path.join(args.dir_model, 'options.txt')
        dict_opt = vars(args)

        if args.is_train:
            with open(args.file_log, 'wt') as log:
                print("-" * 50, 'Options', "-" * 50)
                for k, v in dict_opt.items():
                    print("{}: {}".format(k, v))
                    log.write('{}, {}'.format(k, v))
                print("-" * 100)

        return args


class TrainOption(BaseOption):
    def __init__(self):
        super(TrainOption, self).__init__()
        self.args.add_argument('--is_train', action='store_true', default=True)

        self.args.add_argument('--beta_1', type=float, default=0.5, help='Adam optimizer param.')
        self.args.add_argument('--beta_2', type=float, default=0.999, help='Adam optimizer param.')
        self.args.add_argument('--iter_display', type=int, default=1, help='frequency you want to see training images'
                                                                           'in iteration.')
        self.args.add_argument('--iter_report', type=int, default=1, help='frequency you want to be reported losses.')
        self.args.add_argument('--iter_val', type=int, default=1, help='frequency you want to see validation images'
                                                                        'in iteration.')
        self.args.add_argument('--epoch_decay', type=int, default=100, help='epoch where learning rate starts to'
                                                                            ' decay.')
        self.args.add_argument('--epoch_save', type=int, default=1, help='frequency you want to save models in epoch.')
        self.args.add_argument('--lambda_cycle', type=int, default=10, help='weight for cycle consistency loss.')
        self.args.add_argument('--lr', type=float, default=2e-4, help='Initial learning rate.')
        self.args.add_argument('--n_buffer_images', type=int, default=50, help='how many images the model stores for'
                                                                               'discriminator updates')
        self.args.add_argument('--n_epochs', type=int, default=100)
        self.args.add_argument('--val_during_training', action='store_true', default=True)


class TestOption(BaseOption):
    def __init__(self):
        super(TestOption, self).__init__()
        self.args.add_argument('--is_train', action='store_true', default=False)
