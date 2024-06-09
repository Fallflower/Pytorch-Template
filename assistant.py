import argparse
import json
import os
import time


class Options():
    """Class to manage options using parser and namespace.
    """

    def __init__(self):
        self.initialized = False

    def add_arguments_parser(self, parser: argparse.ArgumentParser):
        """Add a set of arguments to the parser.
        Parameters
        ----------
        parser : argparse.ArgumentParser
            Parser to add arguments to.
        Returns
        -------
        parser : argparse.ArgumentParser
            Parser with added arguments.
        """
        # parser.add_argument('--batch_size', type=int, default=32, help="input batch size,default = 64")
        # parser.add_argument('--num_images', type=int, default=1024,
        #                     help="number of images, default = 1024, if -1, use all images")
        # parser.add_argument("--ds_enhance", type=bool, default=False, help="decide to enhance dataset(reverse)")
        # parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for, default=10')
        # # parser.add_argument("--seed", type=int, default=66, help="random seed")
        # # parser.add_argument("--log_path", type=str, default="./runs/log")
        # parser.add_argument("--img_size", type=tuple, default=(224, 224), help="input image size")
        # parser.add_argument("--growth_rate", type=int, default=32, help="")
        # parser.add_argument("--block_config", type=tuple, default=(6, 12, 24, 16))
        # parser.add_argument("--num_init_features", type=int, default=64, help="")
        # parser.add_argument("--bn_size", type=int, default=4, help="batchnorm size")
        # parser.add_argument("--drop_rate", type=float, default=0, help="be used to regulate overfitting")
        # parser.add_argument("--num_classes", type=int, default=14, help="classification category,default 10")
        # parser.add_argument("--model", type=str, default="DenseNet", help="")
        # parser.add_argument("--learning_rate", type=float, help="")
        # parser.add_argument("--lr_gamma", type=float, default=0.95, help="learning rate scheduler: lr *= gamma")
        # parser.add_argument("--device", type=str, default="cuda:0", help="")
        # parser.add_argument("--show", action="store_true", default=False)
        self.initialized = True

        return parser

    def _initialize_options(self):
        """Initialize a namespace that store options.
        Returns
        -------
        opt: argparse.Namespace
            Namespace with options.
        """
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.add_arguments_parser(parser)
            self.parser = parser

        else:
            print("WARNING: Options was already initialized before")

        return self.parser.parse_args()

    def parse(self):
        """Initialize a namespace that store options.
        Returns
        -------
        opt: argparse.Namespace
            Namespace with options.
        """
        opt = self._initialize_options()
        return opt

    def print_options(self, opt: argparse.Namespace):
        """Print all options and the default values (if changed).
        Parameters
        ----------
        opt : argparse.Namespace
            Namespace with options to print.
        """
        # create a new parser with default arguments
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.add_arguments_parser(parser)

        message = 'Options: \n'
        for key, value in sorted(vars(opt).items()):
            comment = ''
            default = parser.get_default(key)
            if value != default:
                comment = f'(default {default})'
            key, value = str(key), str(value)
            message += f'\t{key}: {value} {comment}\n'
        print(message)
        return message

    def save_options(self, opt: argparse.Namespace, path: str):
        """Save options to a json file.
        Parameters
        ----------
        opt : argparse.Namespace
            Namespace with options to save.
        path : str
            Path to save the options (.json extension will
            be automatically added at the end if absent).
        """
        if not path.endswith('.json'):
            path += '.json'
        with open(path, 'w') as f:
            f.write(json.dumps(vars(opt), indent=4))

    def load_options(self, path: str):
        # bug:这玩意当你的required = True，你必须输入必需参数，不过load的话可以直接load进网络，不要load进入argparse？
        """Load options from a json file.
        Parameters
        ----------
        path : str
            Path to load the options (.json extension will
            be automatically added at the end if absent).
        Returns
        -------
        opt : argparse.Namespace
            Namespace with loaded options.
        """
        if not path.endswith('.json'):
            path += '.json'
        # init a new namespace with default arguments
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        opt = self.add_arguments_parser(parser).parse_args([])

        variables = json.load(open(path, 'r'))
        for key, value in variables.items():
            setattr(opt, key, value)
        print("--------------load options finish----------------")
        return opt

    def load_options_from_dict(self, opt_dict: dict):
        """Load options from a dict.
        Parameters
        ----------
        opt_dict : dict
            Dict with options to load.
        Returns
        -------
        opt : argparse.Namespace
            Namespace with loaded options.
        """
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        opt = self.add_arguments_parser(parser).parse_args([])

        for key, value in opt_dict.items():
            setattr(opt, key, value)
        print("--------------load options finish----------------")
        return opt


class HelpSave:
    def __init__(self, result_dir="", result_root="results/"):
        self.result_root = result_root
        self.result_dir = self.result_root + result_dir

    def init_save_dir(self, opt: argparse.Namespace):
        time_now = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        dir_name = time_now + '/'

        self.result_dir = self.result_root + dir_name

        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        options = Options()
        with open(self.result_dir + "optInfo.txt", 'w', encoding='utf-8') as f:
            f.write(options.print_options(opt))

    def save_train_info(self, info_dict):
        if not os.path.exists(self.result_dir + 'train_info.csv'):
            with open(self.result_dir + 'train_info.csv', 'w', encoding='utf-8') as f:
                f.write(",".join(info_dict.keys()) + "\n")
        with open(self.result_dir + 'train_info.csv', 'a', encoding='utf-8') as mf:
            str_list = []
            for i in info_dict.values():
                if type(i) is int:
                    str_list.append("%d"%i)
                else:
                    str_list.append("%.6f"%i)
            mf.write(",".join(str_list)+"\n")

    def save_test_info(self, info_dict):
        if not os.path.exists(self.result_dir + 'test_info.csv'):
            with open(self.result_dir + 'test_info.csv', 'w', encoding='utf-8') as f:
                f.write(",".join(info_dict.keys()) + "\n")
        with open(self.result_dir + 'test_info.csv', 'a', encoding='utf-8') as mf:
            str_list = []
            for i in info_dict.values():
                if type(i) is int:
                    str_list.append("%d" % i)
                else:
                    str_list.append("%.6f" % i)
            mf.write(",".join(str_list) + "\n")
            