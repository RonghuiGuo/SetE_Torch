import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from configs.SetE import get_arguments
from utils.loss_fn import SetE_loss
from utils.utils import *


def train(args, model, iterator, criterion, optimizer, iter_cnt):
    model.train()

    all_loss = []
    # 进度条
    tqdm_iterator = tqdm(iterator)
    for i, batch in enumerate(tqdm_iterator):
        flag = batch['flag']
        data_pos = [item.to(args.device) for item in batch['data_pos']]
        data_neg = [item.to(args.device) for item in batch['data_neg']]

        # forward
        embeds_pos, embeds_neg = model(flag, data_pos, data_neg)
        loss = criterion(args, flag, embeds_pos, embeds_neg)
        all_loss.append(loss.item())
        # tensorboard+logger
        # args.logger.add_scalar('train/loss', loss.item(), iter_cnt+i)

        # backward
        loss = loss / args.gradient_accumulation  # 梯度积累，GPU放不下一个batch时需要缩减放入GPU的数据量
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            get_trainable_parameters(model), max_norm=args.max_grad_norm)

        # optimize
        if (i + 1) % args.gradient_accumulation == 0 or (i + 1) == len(tqdm_iterator):
            optimizer.step()
            optimizer.zero_grad()

        train_loss = np.mean(all_loss)
        train_ppl = math.exp(train_loss)  # 这里应该有点问题，要改一改
        tqdm_iterator.set_description(
            f'Loss: {train_loss:.3f} | PPL: {train_ppl:.3f} |')

    ppl = math.exp(np.mean(all_loss))
    return {'PPL': ppl}


def start_train(args):
    # 保存超参数
    save_args_to_yaml(args)

    # 读入数据
    args.logger.debug(f'[*] Loading dataset: {args.dataset} ...')
    # train_iterator, valid_iterator, test_iterator = args.get_dataloaders(args.batch_size, skip_list=[1, 2])

    # 构建模型
    args.logger.debug(f'[*] Building model: {args.model_name} ...')
    model = args.network(args).to(args.device)

    # 权重初始化
    model.init_weights()
    show_network(args, model, args.model_name)

    # 损失函数与优化器
    criterion = SetE_loss
    optimizer = optim.Adam(get_trainable_parameters(
        model), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)

    # 从ckpt恢复训练
    if args.train_from_ckpt:
        args.logger.debug(
            f'[*] Loading model from {args.ckpt_path}, train start ...')
        checkpoint = torch.load(args.ckpt_path)
        model.load_state_dict(checkpoint['model'])
        best_epoch, best_iter, best_score = checkpoint['epoch'], checkpoint['iter'], checkpoint['score']
        start_epoch, start_iter = best_epoch-1, best_iter
        # scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        args.logger.debug(f'[*] Train model from scratch, train start ...')
        best_epoch, best_iter, best_score = 0, 0, float('inf')
        start_epoch, start_iter = best_epoch, best_iter

    # 开始训练
    early_stopping_cnt = 0
    for epoch in range(start_epoch, args.epochs):
        args.logger.debug(f'Epoch: {epoch+1:02}/{args.epochs:02} ...')

        # 读入数据
        train_iterator, valid_iterator, test_iterator = args.get_dataloaders(
            args.batch_size, skip_list=[1, 2])

        all_batch = train_iterator
        train_batch_list, train_batch_list_len = [all_batch], [len(all_batch)]

        for i in range(start_iter, len(train_batch_list)):
            # 训练/验证步数
            tarin_iter_cnt = epoch * \
                sum(train_batch_list_len) + sum(train_batch_list_len[:i])
            valid_iter_cnt = epoch*len(train_batch_list)

            # 训练模型
            train_loss_dict = train(
                args, model, train_batch_list[i], criterion, optimizer, tarin_iter_cnt)

            # 记录权重
            # args.logger.add_trainable_parameters(model, valid_iter_cnt+i, args.model_name)
            args.logger.debug(f'    [{epoch+1:02}-{i+1:02} - {get_time()}]' +
                              f' [Train] {convert_dict_to_string(train_loss_dict)}')

            # 早停机制
            current_score = train_loss_dict['PPL']
            if current_score < best_score:
                # 保存ckpt
                # checkpoint = {'model': model.state_dict(), 'scheduler': scheduler.state_dict(),
                #               'epoch': epoch+1, 'iter': i+1, 'score': current_score}
                checkpoint = {'model': model.state_dict(
                ), 'epoch': epoch+1, 'iter': i+1, 'score': current_score}
                # torch.save(checkpoint, f'{args.ckpt_dir}/ckpt_{epoch+1:02}-{i+1:02}.pth')
                torch.save(checkpoint, f'{args.ckpt_dir}/ckpt_best.pth')

                early_stopping_cnt, best_score, best_epoch, best_iter = 0, current_score, epoch+1, i+1
                args.logger.debug(
                    f'  Save model trained at {epoch+1:02}-{i+1:02}!')
            else:
                early_stopping_cnt += 1
                args.logger.debug(f'  EarlyStopping counter: {early_stopping_cnt} out of ' +
                                  f'{args.early_stopping_patience} at {epoch+1:02}-{i+1:02}!')
                if early_stopping_cnt >= args.early_stopping_patience:
                    break

        start_iter = 0  # 从ckpt恢复训练时初始化一次i,后续的i从0开始
        if early_stopping_cnt >= args.early_stopping_patience:
            break

    args.logger.info(f'[*] The model has been trained successfully! ' +
                     f'The best_score is {best_score:.3f} at {best_epoch:02}-{best_iter:02}it!')


if __name__ == '__main__':
    # 开始计时
    start_time = time.time()

    # 参数解析
    args = get_arguments()
    show_running_status(args)

    # 设置随机种子
    set_random_seed(args.seed)

    try:
        # 运行程序
        # globals() 函数会以字典类型返回当前位置的全部全局变量。
        globals()[f'start_{args.mode}'](args)
    except KeyboardInterrupt:
        # Ctrl+C
        args.logger.error(f'[*] KeyboardInterrupt, {args.mode} interrupted!')
    finally:
        # 结束计时
        end_time = time.time()
        show_running_time(args, start_time, end_time)
