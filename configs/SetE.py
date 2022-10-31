import argparse
import importlib
import os
import shutil
import torch
from utils.logger import Logger
from utils.utils import load_args_from_yaml


def get_arguments():
    parser = argparse.ArgumentParser()

    # 解析选择的模型与数据集
    parser.add_argument('--run_id', type=str,
                        default='debug', help='实验组id,用于区分不同的参数配置')
    parser.add_argument('--model_name', type=str,
                        default='SetE', choices=['SetE'])
    parser.add_argument('--dataset', type=str,
                        default='YAGO39K', choices=['YAGO39K'])
    # parser.add_argument('--mode', type=str,required=True, choices=['train', 'test'])
    parser.add_argument('--mode', type=str, default="train",
                        choices=['train', 'test'])
    # 只返回知道的参数
    args, _ = parser.parse_known_args()

    # 解析公有的命令行参数
    parser = get_parser_all_mode_common(parser)
    args, _ = parser.parse_known_args(namespace=args)

    # 配置ckpt/args/logs/logging路径
    # checkpoints保存目录
    # f 格式化字符串常量
    args.ckpt_dir = f'checkpoints/{args.model_name}/{args.run_id}'
    args.args_path = f'{args.ckpt_dir}/args.yaml'  # args保存路径
    # tensorboard_logs保存目录
    args.logs_dir = f'logs/{args.model_name}/{args.run_id}'
    # logging_logs保存路径
    args.logging_path = f'{args.logs_dir}/{args.mode}_log.txt'

    if args.mode == 'train':
        # 解析剩下的命令行参数
        # 调用globals()函数，返回当前全局符号表的字典
        parser = globals()[f'get_parser_train_mode'](parser)
        # 与上面那句话等价
        # parser = get_parser_train_mode(parser)

        args = parser.parse_args(namespace=args)

        # 如果从零开始重新训练,则删除之前的ckpt/log文件
        if not args.train_from_ckpt:
            if os.path.exists(args.ckpt_dir):
                # 递归删除文件夹
                shutil.rmtree(args.ckpt_dir)
            if os.path.exists(args.logs_dir):
                shutil.rmtree(args.logs_dir)

        # 如果恢复训练使用的ckpt不是最新的,则删除旧ckpt文件
        if args.train_from_ckpt and args.ckpt_id != None:
            ckpt_list = os.listdir(args.ckpt_dir)[1:]
            for ckpt in ckpt_list:
                ckpt_id = ckpt.replace('ckpt_', '').replace('.pth', '')
                if ckpt_id > args.ckpt_id:
                    os.remove(f'{args.ckpt_dir}/{ckpt}')

        # 如果从ckpt中恢复训练且没有指定ckpt,则选择最新的ckpt
        if args.train_from_ckpt and args.ckpt_id == None:
            latest_ckpt = sorted(os.listdir(args.ckpt_dir))[-1]
            args.ckpt_id = latest_ckpt.replace('ckpt_', '').replace('.pth', '')

    elif args.mode == 'test':
        # 如果没有指定测试使用的ckpt,则选择最新的ckpt
        if args.ckpt_id == None:
            latest_ckpt = sorted(os.listdir(args.ckpt_dir))[-1]
            args.ckpt_id = latest_ckpt.replace('ckpt_', '').replace('.pth', '')

        # 从args_path中加载已经保存的模型训练参数
        args = load_args_from_yaml(
            args, yaml_path=args.args_path, skip_list=list(args.__dict__.keys()))

    # 创建ckpt/args/logs/logging目录，如果已经存在则不创建
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)

    # checkpoints保存路径
    args.ckpt_path = f'{args.ckpt_dir}/ckpt_{args.ckpt_id}.pth'

    # 导入模型
    # import_model module models.SetE
    # args.network class models.SetE.SetE
    import_model = importlib.import_module(f'models.{args.model_name}')
    args.network = getattr(import_model, args.model_name)

    # 导入数据集
    import_data = importlib.import_module(f'datasets.{args.dataset}')
    args.get_dataloaders = getattr(import_data, 'get_dataloaders')

    # 配置运行环境
    args.device = torch.device(
        'cuda' if not args.use_cpu and torch.cuda.is_available() else 'cpu')

    # 创建logger
    args.logger = Logger(args)

    return args


def get_parser_all_mode_common(parser):
    parser.add_argument('--stdout_handler_level', type=str, default='debug',
                        choices=['debug', 'info', 'notice', 'warning', 'error', 'critical'], help='标准输出的log输出级别')
    parser.add_argument('--file_handler_level', type=str, default='debug',
                        choices=['debug', 'info', 'notice', 'warning', 'error', 'critical'], help='log文件的log输出级别')
    parser.add_argument('--seed', type=int, default=1895,
                        help='设置随机数种子,保证模型的可复现性')
    parser.add_argument('--use_cpu', action='store_true', help='是否使用CPU进行训练')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--ckpt_id', type=str,
                        default=None, help='恢复模型所需的ckpt的id')
    return parser


def get_parser_train_mode(parser):
    parser.add_argument('--train_from_ckpt',
                        action='store_true', help='训练时是否从ckpt中恢复训练')
    parser.add_argument('--epochs', type=int, default=2,
                        help='请限制在2位数或修改ckpt_id相关代码,否则恢复训练时可能出现bug')
    parser.add_argument('--train_iter_step', type=int,
                        default=-1, help='每训练多少个batch测试一次')
    parser.add_argument('--early_stopping_patience',
                        type=int, default=5, help='早停机制')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--gradient_accumulation',
                        type=int, default=1, help='梯度积累')
    parser.add_argument('--max_grad_norm',  type=float,
                        default=1.0, help='梯度裁断')

    parser.add_argument('--instance_num', type=int,
                        default=39374, help='实例的个数')
    parser.add_argument('--concept_num', type=int, default=46109, help='概念的个数')
    parser.add_argument('--relation_num', type=int, default=37, help='关系的个数')
    parser.add_argument('--emb_dim', type=int, default=50, help='表示的维数')

    parser.add_argument('--B_t', type=float, default=1, help='表示的维数')
    parser.add_argument('--B_r', type=float, default=2, help='表示的维数')
    return parser
