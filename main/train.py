import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import argparse
import sys
import os.path as osp

# Add project root and mmcv to path
project_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, osp.join(project_root, 'mmcv'))

import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from config import cfg
# ddp
import torch.distributed as dist
from common.utils.distribute_utils import (
    init_distributed_mode, is_main_process, set_seed
)
import torch.distributed as dist
from mmcv.runner import get_dist_info
from common.utils.check_dataload import show_input_image
from dataset import print_dataset_info


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--num_gpus', type=int, dest='num_gpus')
    parser.add_argument('--master_port', type=int, dest='master_port')
    parser.add_argument('--exp_name', type=str, default='output/test')
    parser.add_argument('--config', type=str, default='./config/config_base.py')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    config_path = osp.join('./config', args.config)
    cfg.get_config_fromfile(config_path)

    cudnn.benchmark = True
    set_seed(2023)

    # ddp by default in this branch
    distributed, gpu_idx = init_distributed_mode(args.master_port)

    # Only rank 0 creates directories, then broadcast the output_dir to all ranks
    if distributed:
        if is_main_process():
            cfg.update_config(args.num_gpus, args.exp_name)
            output_dir = cfg.output_dir
        else:
            output_dir = None

        # Broadcast output_dir from rank 0 to all ranks
        import torch.distributed as dist
        output_dir_list = [output_dir]
        dist.broadcast_object_list(output_dir_list, src=0)
        output_dir = output_dir_list[0]

        # Non-rank-0 processes update their config with the shared output_dir
        if not is_main_process():
            cfg.num_gpus = args.num_gpus
            cfg.exp_name = args.exp_name
            cfg.output_dir = output_dir
            cfg.model_dir = osp.join(output_dir, 'model_dump')
            cfg.vis_dir = osp.join(output_dir, 'vis')
            cfg.log_dir = osp.join(output_dir, 'log')
            cfg.code_dir = osp.join(output_dir, 'code')
            cfg.result_dir = osp.join(output_dir, 'result')

        # Ensure all ranks wait for rank 0 to finish directory creation
        dist.barrier()
    else:
        cfg.update_config(args.num_gpus, args.exp_name)
    from base import Trainer
    trainer = Trainer(distributed, gpu_idx)
    
    # Fixed by SH Heo (251230) - Support single-GPU training without DDP
    if distributed:
        trainer.logger_info('### Set DDP ###')
        trainer.logger.info(f'Distributed: {distributed}, init done {gpu_idx}')
    else:
        trainer.logger_info('### Single GPU Mode (No DDP) ###')
        trainer.logger.info(f'Distributed: {distributed}, GPU idx: {gpu_idx}')
    
    trainer.logger_info(f"Using {cfg.num_gpus} GPUs, batch size {cfg.train_batch_size} per GPU.")
    
    trainer._make_batch_generator()
    trainer._make_model()

    # Fixed by SH Heo (260108) - Print dataset info
    if is_main_process():
        if trainer.train_dataset_info:
            print_dataset_info(trainer.train_dataset_info, 'Train')
        if hasattr(trainer, 'valid_dataset_info') and trainer.valid_dataset_info:
            print_dataset_info(trainer.valid_dataset_info, 'Valid')

    trainer.logger_info('### Start training ###')

    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        trainer.tot_timer.tic()
        trainer.read_timer.tic()
        
        # ddp, align random seed between devices
        if distributed:
            trainer.batch_generator.sampler.set_epoch(epoch)

        # tqdm progress bar (only show on main process)
        if is_main_process():
            pbar = tqdm(total=trainer.itr_per_epoch,
                       desc=f'Epoch {epoch}/{cfg.end_epoch}',
                       ncols=120)

        for itr, (inputs, targets, meta_info) in enumerate(trainer.batch_generator):
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()

            # forward
            trainer.optimizer.zero_grad()
            show_input_image(inputs)
            loss= trainer.model(inputs, targets, meta_info, 'train')

            loss_mean = {k: loss[k].mean() for k in loss}
            loss_sum = sum(loss_mean[k] for k in loss_mean)
            
            # backward
            loss_sum.backward()
            trainer.optimizer.step()
            trainer.scheduler.step()
            
            trainer.gpu_timer.toc()

            # loss of all ranks
            loss_print = loss_mean.copy()
            if distributed:
                rank, world_size = get_dist_info()
                for k in loss_print:
                    dist.all_reduce(loss_print[k])
                for k in loss_print:
                    loss_print[k] = loss_print[k] / world_size

            total_loss = sum(loss_print[k] for k in loss_print)
            loss_print['total'] = total_loss

            # update tqdm progress bar
            if is_main_process():
                pbar.set_postfix({
                    'lr': f"{trainer.get_lr():.2e}",
                    'loss': f"{loss_print['total'].item():.4f}",
                    'speed': f"{trainer.tot_timer.average_time:.2f}s/it"
                })
                pbar.update(1)

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()

        # close tqdm progress bar
        if is_main_process():
            pbar.close()

        # Save train loss for TensorBoard logging (last iteration value)
        train_loss_for_log = {k: v.item() for k, v in loss_print.items()}

        # # epoch summary log
        # screen = [
        #     'Epoch %d/%d:' % (epoch, cfg.end_epoch),
        #     'lr: %g' % (trainer.get_lr()),
        #     'speed: %.2f(%.2fs r%.2f)s/itr' % (
        #         trainer.tot_timer.average_time, trainer.gpu_timer.average_time,
        #         trainer.read_timer.average_time),
        #     '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
        # ]
        # screen += ['%s: %.4f' % ('loss_' + k, v.detach()) for k, v in loss_print.items()]
        # trainer.logger_info(' '.join(screen))

        # Validation
        valid_loss_avg = None
        if trainer.valid_batch_generator is not None:
            trainer.model.eval()
            valid_loss_sum = {}
            valid_count = 0

            with torch.no_grad():
                if is_main_process():
                    valid_pbar = tqdm(trainer.valid_batch_generator, desc='Valid', leave=True)
                else:
                    valid_pbar = trainer.valid_batch_generator

                for inputs, targets, meta_info in valid_pbar:
                    loss = trainer.model(inputs, targets, meta_info, 'train')
                    loss_mean = {k: v.mean() for k, v in loss.items()}

                    for k, v in loss_mean.items():
                        if k not in valid_loss_sum:
                            valid_loss_sum[k] = 0.0
                        valid_loss_sum[k] += v.item()
                    valid_count += 1

            # Average validation loss
            valid_loss_avg = {k: v / valid_count for k, v in valid_loss_sum.items()}
            valid_loss_avg['total'] = sum(valid_loss_avg.values())

            # Log validation loss
            valid_screen = ['[Valid]'] + ['%s: %.4f' % ('loss_' + k, v) for k, v in valid_loss_avg.items()]
            trainer.logger_info(' '.join(valid_screen))

            trainer.model.train()

        # TensorBoard logging (only on main process)
        if is_main_process() and trainer.writer is not None:
            # Log learning rate
            trainer.writer.add_scalar('lr', trainer.get_lr(), epoch)
            # Log epoch time
            trainer.writer.add_scalar('epoch_time_hours',
                                     trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch,
                                     epoch)
            # Log losses (train/valid on same graph)
            for k, v in train_loss_for_log.items():
                if valid_loss_avg is not None and k in valid_loss_avg:
                    trainer.writer.add_scalars(f'loss_{k}', {
                        'train': v,
                        'valid': valid_loss_avg[k]
                    }, epoch)
                else:
                    trainer.writer.add_scalars(f'loss_{k}', {'train': v}, epoch)

        # save model ddp, save model.module on rank 0 only
        save_epoch = getattr(cfg, 'save_epoch', 10)
        if is_main_process() and (epoch % save_epoch == 0 or epoch == cfg.end_epoch - 1):
            trainer.save_model({
                'epoch': epoch,
                'network': trainer.model.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
            }, epoch)

        if distributed:
            dist.barrier()

if __name__ == "__main__":
    main()