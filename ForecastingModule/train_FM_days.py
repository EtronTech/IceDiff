from Model.SwinTrV2_UNet import SwinV2_Unet

from Data.dataloader_week_batch import build_dataloader # 7 days
from Learning.utils import set_random_seed, train_one_iteration, validate_training, save_checkpoint, \
                            resume_checkpoint, logger, MetricLogger 
from Learning.lr_scheduler import build_scheduler
from Learning.optimizer import build_optimizer
import logging
import ConfigHandler as cfg
import torch
import torch.cuda.amp as amp
from tqdm import tqdm
import time
from datetime import datetime, timedelta
from torch.utils.tensorboard import SummaryWriter
import os
import gc

import argparse

def icediff_FM_training(config_file=None, args = None,resume=False, gpu_rank = 0):

    config_file = config_file

    task = 'IceDiff_FM'

    config = cfg.Handler(config_file).get_data
    conf = config.PRED_FOR_SR

    set_random_seed(conf.TRAIN.RANDOM_SEEDS)

    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True

    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True

    # Optimize Convolution Operator
    torch.backends.cudnn.benchmark = True

    torch.cuda.set_device(gpu_rank)

    timestamp = time.strftime('%m-%d_%H-%M-%S')
    run_name = '{:s}.{:s}'.format(task,timestamp)

    os.makedirs(conf.TRAIN.LOGGING_DIR, exist_ok=True)
    train_logger = logger('IceDiff', conf.TRAIN.LOGGING_DIR, comment='{:s}'.format(run_name))
    train_logger.info('{:d} GPUs available'.format(torch.cuda.device_count()))
    train_logger.info('Runtime args :')
    train_logger.info(args)
    train_logger.info('Loaded configuration file {:s}'.format(config_file))

   
    ckpt = resume_checkpoint(conf.TRAIN.SAVE_CKPT_DIR)
    
    model = SwinV2_Unet(config)

    en,de, overall = model.count_parameters()
    # print("Encoder has Num of Params: ",en)
    # print("Decoder has Num of Params: ",de)
    # print("The Model has Num of Params: ",overall)

    train_log = logging.getLogger('IceDiff.train')
    train_log.info("Encoder has Num of Params: {:f} Million".format(en))
    train_log.info("Decoder has Num of Params: {:f} Million".format(de))
    train_log.info("The Model has Num of Params: {:f} Million".format(overall))


    model = model.cuda()
    
    train_log.info('Setup Training Environment...')
    
    # Setup DataLoader
    start_iteration = 0
    if ckpt is not None:
        start_iteration = ckpt['iter'] + 1
        train_log.info('Starts from ckpt iter: {:d}'.format(start_iteration))
    else:
        train_log.info('Starts from ckpt iter: 0')
    train_dataloader, max_iteration, n_iters_per_epoch = build_dataloader('TRAIN',task,
                                  config.DATA_SETS.SIC.OUTPUT.DIR,
                                  start_iteration,
                                  config.PRED_FOR_SR.TRAIN.EPOCHS,
                                  config.PRED_FOR_SR.TRAIN.BATCH_SIZE,
                                  config.PRED_FOR_SR.TRAIN.NUM_WORKERS)
    
    val_dataloader, _, _ = build_dataloader('VAL',task,
                                  config.DATA_SETS.SIC.OUTPUT.DIR,
                                  0,
                                  config.PRED_FOR_SR.TRAIN.EPOCHS,
                                  1,
                                  config.PRED_FOR_SR.TRAIN.NUM_WORKERS)

    # Setup Optimizer
    optimizer = build_optimizer(config.PRED_FOR_SR,
                                model)
    clip_grad = conf.TRAIN.OPTIMIZER.CLIP_GRAD

    # Setup Scheduler
    lr_scheduler = build_scheduler(config.PRED_FOR_SR,
                                   optimizer=optimizer,
                                   n_iter_per_epoch=len(train_dataloader))

    # Setup Scalar 
    scaler = amp.GradScaler(enabled=True)

    # Setup Tensorboard
    tensor_board_record = SummaryWriter(conf.TRAIN.TENSOR_BOARD_DIR)
    train_metric_logger = MetricLogger('Train',conf.TRAIN.LOGGING_DIR)
    val_metric_logger = MetricLogger('Val',conf.TRAIN.LOGGING_DIR)

    best_metric_mse = 99.0
    best_metric_mae = 99.0
    best_metric_mse_EPOCH = 0
    best_metric_mae_EPOCH = 0
    validation_epoch = 0
    if ckpt is not None:
        train_log.info('Load Previous Checkpoint...')
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        model.load_state_dicts(ckpt['model'])
        best_metric = ckpt['record_metrics']
        validation_epoch = (ckpt['iter']//n_iters_per_epoch + 1)//conf.TRAIN.VAL_PER_EPOCH
        train_log.info('previous validation_epoch = {:d}'.format(validation_epoch))

        best_metric_mse = best_metric['MSE']
        best_metric_mae = best_metric['MAE']
        

    # Train with Iteration based Sampler
    start_time = time.time()
    model.train()
    train_log.info('Enumerate train_dataloader...')
    __dataloader__ = enumerate(train_dataloader)
    train_log.info('Train with Iteration based Sampler...')
    with tqdm(total=max_iteration-start_iteration) as bar:
        for iter_i in range(start_iteration,max_iteration):
            _, batch = __dataloader__.__next__()
            train_one_iteration(model, batch, optimizer, iter_i, lr_scheduler, scaler, train_metric_logger,clip_grad)
            bar.update(1)
            
            # Evaluate and Save Checkpoint
            equivalent_epoch = iter_i//n_iters_per_epoch + 1 # Epoch starts from 1
            if equivalent_epoch//conf.TRAIN.VAL_PER_EPOCH - validation_epoch > 0 or iter_i == max_iteration:
                '''
                Validation Records
                '''
                validation_epoch += 1
                val_log = logging.getLogger('IceDiff.Val')
                val_log.info('Start Validation...')
                validate_training(model,
                                   val_dataloader,
                                   val_metric_logger)
                
                val_metric_logger.save_log(overwrite=False,global_iter=str(iter_i))

                avg_mse = val_metric_logger.get_data['mse_loss_trace']['average']
                avg_mae = val_metric_logger.get_data['mae_loss_trace']['average']
                if avg_mse < best_metric_mse:
                    best_metric_mse = avg_mse
                    best_metric_mse_EPOCH = equivalent_epoch
                    val_log.info('Update New Best MSE at: Epoch-{:d}, Iteration-{:d}'.format(equivalent_epoch,iter_i))
                if avg_mae < best_metric_mae:
                    best_metric_mae = avg_mae
                    best_metric_mae_EPOCH = equivalent_epoch
                    val_log.info('Update New Best MAE at: Epoch-{:d}, Iteration-{:d}'.format(equivalent_epoch,iter_i))

                val_log.info('Finished Validation. MSE: {:f}, MAE: {:f}'.format(avg_mse,avg_mae))
                val_log.info('SIE Aera Diff: {:f}'.format(val_metric_logger.get_data['sie_regularization_area_trace']['average']))
                val_log.info('SIE IoU: {:f}'.format(val_metric_logger.get_data['sie_regularization_iou_trace']['average']))
                val_log.info('Recorded Best MSE: {:f}, at: Epoch-{:d}'.format(best_metric_mse,best_metric_mse_EPOCH))
                val_log.info('Recorded Best MAE: {:f}, at: Epoch-{:d}'.format(best_metric_mae,best_metric_mae_EPOCH))

                model_state_dicts = model.get_state_dicts()               
                save_checkpoint(model_state_dicts,
                                config,
                                conf.TRAIN.SAVE_CKPT_DIR,
                                equivalent_epoch,
                                iter_i,
                                {'MSE':best_metric_mse,'MAE':best_metric_mae},
                                optimizer,
                                lr_scheduler,
                                scaler)

                tensor_board_record.add_scalar('val/' + 'mse_average', 
                                               val_metric_logger.get_data['mse_loss_trace']['average'], 
                                               global_step=iter_i)
                tensor_board_record.add_scalar('val/' + 'mae_average', 
                                               val_metric_logger.get_data['mae_loss_trace']['average'], 
                                               global_step=iter_i)
                
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0 * 1024)
                val_log.info('Max GPU Memory Allocated: {:f} GB'.format(memory_used))
                val_log.info('Resume Training...')

                # Early Stop
                if conf.TRAIN.EARLY_STOP_EPOCH == equivalent_epoch:
                    val_log.info('Early Stop at Epoch: {:d}'.format(equivalent_epoch))
                    # total_time = time.time() - start_time
                    # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
                    # print('Total Training Time Elapsed: ', total_time_str)
                    return

            else:
                '''
                Training Records
                '''
                if iter_i%conf.TRAIN.RECORD_PER_BATCH == 0:
                    mse_error = train_metric_logger.get_data['mse_loss_trace']['average']
                    tensor_board_record.add_scalar('train/' + 'mse_average', 
                                            mse_error, 
                                            global_step=iter_i)
                    mae_error = train_metric_logger.get_data['mae_loss_trace']['average']
                    tensor_board_record.add_scalar('train/' + 'mae_average', 
                                            mse_error, 
                                            global_step=iter_i)
                    memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0 * 1024)
                    train_log.info('Max GPU Memory Allocated: {:f} GB'.format(memory_used))
                    train_log.info('RECORD Average Gradient Norm: {:f}'.format(train_metric_logger.get_data['grad_norm']))
                    train_log.info('RECORD Average Mse Loss: {:f}'.format(mse_error))
                    train_log.info('RECORD Average Mae Loss: {:f}'.format(mae_error))
                
                if iter_i%conf.TRAIN.SAVE_CKPT_PER_BATCH == 0:
                    # Save Chekpoint
                    model_state_dicts = model.get_state_dicts()               
                    save_checkpoint(model_state_dicts,
                                    config,
                                    conf.TRAIN.SAVE_CKPT_DIR,
                                    equivalent_epoch,
                                    iter_i,
                                    {'MSE':best_metric_mse,'MAE':best_metric_mae},
                                    optimizer,
                                    lr_scheduler,
                                    scaler)
    




if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_rank', type=int,
                    default=0, help='designate gpu rank.')
    
    parser.add_argument('--conf_pth', type=str,
                    default='None', help='Configuration file.')        


    args = parser.parse_args()
    
    gpu_rank = args.gpu_rank

    config_file = args.conf_pth

    icediff_FM_training(config_file=config_file,gpu_rank=gpu_rank)