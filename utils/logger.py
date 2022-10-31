import logging
import sys
import colorlog
from tensorboardX import SummaryWriter


class Logger():
    def __init__(self, args):
        # SummaryWriter的Placeholder
        self.writer = None
        self.writer_logdir = args.logs_dir

        # 创建一个logging_logger实例
        self.set_logging_logger(args)

    def set_writer(self):
        if self.writer == None:
            self.writer = SummaryWriter(
                logdir=self.writer_logdir, purge_step=1, flush_secs=30)

    def add_scalar(self, tag, scalar_value, global_step):
        self.set_writer()
        self.writer.add_scalar(tag, scalar_value, global_step)

    def add_graph(self, model, input_to_model):
        self.set_writer()
        self.writer.add_graph(model, input_to_model)

    def add_histogram(self, tag, values, global_step):
        self.set_writer()
        self.writer.add_histogram(tag, values, global_step)

    def add_trainable_parameters(self, model, epoch, model_name):
        for name, param in model.named_parameters():
            if param.requires_grad:
                if name.startswith(model_name):
                    self.add_histogram(name.replace(
                        '.', '/', 1), param.data, epoch)
                else:
                    self.add_histogram(
                        f'{model_name}/{name}', param.data, epoch)

    def set_logging_logger(self, args):
        # DEBUG(10) < INFO(20) < NOTICE(25) < WARNING(30) < ERROR(40) < CRITICAL(50)
        logging.NOTICE = 25  # 自定义log等级
        logging.addLevelName(logging.NOTICE, 'NOTICE')

        self.logging_logger = logging.getLogger(
            name=args.model_name)  # 创建一个logging_logger实例，初始化
        # 消息先经过logger过滤,后经过handler过滤
        self.logging_logger.setLevel(logging.DEBUG)

        if not self.logging_logger.handlers:  # 防止重复实例化Logger时重复打印
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setLevel(
                getattr(logging, args.stdout_handler_level.upper()))

            file_handler = logging.FileHandler(args.logging_path, mode='a')
            file_handler.setLevel(
                getattr(logging, args.file_handler_level.upper()))

            log_colors_config = {
                'INFO': 'green',
                'NOTICE': 'blue',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red'
            }
            stdout_formatter = colorlog.ColoredFormatter(fmt='%(log_color)s%(message)s',
                                                         log_colors=log_colors_config)
            stdout_handler.setFormatter(
                stdout_formatter)  # stdout_handler设置输出格式

            file_formatter = logging.Formatter('%(message)s')
            file_handler.setFormatter(file_formatter)

            self.logging_logger.addHandler(stdout_handler)
            self.logging_logger.addHandler(file_handler)

        self.set_logging_fn()

    def set_logging_fn(self):  # 将logging_logger的方法引用到Logger中
        fn_list = ['debug', 'info', 'warning']
        for fn in fn_list:
            setattr(self, fn, getattr(self.logging_logger, fn))

    def error(self, msg, *args, **kwargs):
        self.logging_logger.error(msg, *args, **kwargs)
        sys.exit()

    def critical(self, msg, *args, **kwargs):
        self.logging_logger.critical(msg, *args, **kwargs)
        sys.exit()

    def notice(self, msg, *args, **kwargs):  # 自定义新的logging_fn
        if self.logging_logger.isEnabledFor(logging.NOTICE):
            self.logging_logger._log(logging.NOTICE, msg, args, **kwargs)
