import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import argparse
import time
import sys
import os
import logging

from collections import defaultdict


curr_path = os.path.abspath(__file__)
curr_path = curr_path.split('/')[:-2]
curr_path = '/'.join(curr_path)
sys.path.append(curr_path)


from sgan.data.dataloader import data_loader
from sgan.utils import get_dataset_path, relative_to_abs, get_total_norm, int_tuple, bool_flag
from sgan.models import TrajectoryGenerator, TrajectoryDiscriminator
from sgan.losses import l2_loss, displacement_error, final_displacement_error, g_loss, d_loss


torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser()
FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
logging.basicConfig(level = logging.INFO, format = FORMAT, stream = sys.stdout)
logger = logging.getLogger(__name__)

# Dataset options
parser.add_argument('--dataset_name', default = 'zara1', type = str)
parser.add_argument('--delim', default = 'tab')
parser.add_argument('--loader_num_workers', default = 4, type = int)
parser.add_argument('--obs_len', default = 8, type = int)
parser.add_argument('--pred_len', default = 12, type = int)
parser.add_argument('--skip', default = 1, type = int)

# Optimization
parser.add_argument('--batch_size', default = 32, type = int)
parser.add_argument('--num_iterations', default = 500, type = int)
parser.add_argument('--num_epochs', default = 500, type = int)

# Model Options
parser.add_argument('--embedding_dim', default = 64, type = int)
parser.add_argument('--num_layers', default = 1, type = int)
parser.add_argument('--dropout', default = 0.0, type = float)
parser.add_argument('--batch_norm', default = 0, type = bool_flag)

# Generator Options
parser.add_argument('--encoder_h_dim_g', default = 64, type = int)
parser.add_argument('--decoder_h_dim_g', default = 128, type = int)
parser.add_argument('--g_learning_rate', default = 5e-4, type = float)
parser.add_argument('--g_steps', default = 1, type = int)
parser.add_argument('--clip_threshold_g', default = 0.0, type = float)
parser.add_argument('--noise_type', default = 'gaussian', type = str)
parser.add_argument('--noise_dim', default = None, type = int_tuple)
parser.add_argument('--noise_mix_type', default = 'ped', type = str)
parser.add_argument('--decoder_mlp_dim', default = 1024, type = int)

# Pooling Options
parser.add_argument('--pooling_type', default = 'pool_net', type = str)
parser.add_argument('--pool_every_step', default = 1, type = bool_flag)

# Pool Net Options
parser.add_argument('--bottleneck_dim', default = 1024, type = int)
parser.add_argument('--pool_mlp_dim', default = 512, type = int)

# Social Pooling Options
parser.add_argument('--neighbourhood_size', default = 2.0, type = float)
parser.add_argument('--grid_size', default = 8, type = int)

# Discriminator Options
parser.add_argument('--d_type', default = 'ped', type = str)
parser.add_argument('--d_steps', default = 2, type = int)
parser.add_argument('--clip_threshold_d', default = 0.0, type = float)
parser.add_argument('--d_learning_rate', default = 5e-4, type = float)
parser.add_argument('--encoder_h_dim_d', default = 64, type = int)

# Loss Options
parser.add_argument('--l2_loss_weight', default = 0.0, type = float)
parser.add_argument('--best_k', default = 1, type = int)

# Output
parser.add_argument('--output_dir', default = os.getcwd())
parser.add_argument('--print_every', default = 50, type = int)
parser.add_argument('--checkpoint_every', default = 500, type = int)
parser.add_argument('--checkpoint_name', default = 'checkpoint', type = str)
parser.add_argument('--checkpoint_start_from', default = None)
parser.add_argument('--restore_from_checkpoint', default = 1, type = int)
parser.add_argument('--num_samples_check', default = 5000, type = int)

# Miscellaneous
parser.add_argument('--use_gpu', default = 1, type = int)
parser.add_argument('--timing', default = 0, type = int)
parser.add_argument('--gpu_num', default = '0', type = str)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
        

def get_dtypes(args):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if args.use_gpu == 1:
        #cuda0 = torch.device('cuda:0')
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    
    return long_dtype, float_dtype


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
    #print(os.environ['CUDA_VISIBLE_DEVICES'])
    #print(torch.cuda.device_count())
    train_path = get_dataset_path(args.dataset_name, 'train')
    val_path = get_dataset_path(args.dataset_name, 'val')
    

    logger.info('Initializing the train set')
    train_dataset, train_loader = data_loader(args, train_path)
    logger.info('Initializing evaluation dataset')
    _, val_loader = data_loader(args, val_path)
    
    iters_per_epoch = len(train_dataset) / args.batch_size / args.d_steps
    if args.num_epochs:
        args.num_iterations = int(iters_per_epoch * args.num_epochs)
        
    logger.info('There are %d iterations per epoch' % iters_per_epoch)
    
    
    # _device_ids = args.gpu_num.split(',')
    # _device_ids = [int(i) for i in _device_ids]
    #print('ids', _device_ids)
    # generator = nn.DataParallel(generator, device_ids = _device_ids)
    #generator = generator.to(f'cuda:{generator.device_ids[0]}')
    #print(next(generator.parameters()).device)
    generator = TrajectoryGenerator(
        embedding_dim = args.embedding_dim,
        encoder_h_dim = args.encoder_h_dim_g,
        decoder_h_dim = args.decoder_h_dim_g,
        num_layers = args.num_layers,
        batch_norm = args.batch_norm,
        obs_len = args.obs_len,
        pred_len = args.pred_len,
        dropout = args.dropout,
        neighbourhood_size = args.neighbourhood_size,
        grid_size = args.grid_size,
        decoder_mlp_dim = args.decoder_mlp_dim,
        bottleneck_dim = args.bottleneck_dim,
        noise_dim = args.noise_dim,
        noise_type = args.noise_type,
        noise_mix_type = args.noise_mix_type,
        pool_type = args.pooling_type,
        pool_every_step = args.pool_every_step,
        pool_mlp_dim = args.pool_mlp_dim
    )
    
    long_dtype, float_dtype = get_dtypes(args)
    
    generator.apply(init_weights)
    generator.type(float_dtype)
    # generator = nn.DataParallel(generator, device_ids = _device_ids)
    #generator = generator.to(f'cuda:{generator.device_ids[0]}')
    #print(next(generator.parameters()).device)
    generator.train()
    logger.info('The generator is demonstrated below: ')
    logger.info(generator)
    
    discriminator = TrajectoryDiscriminator(
        obs_len = args.obs_len,
        pred_len = args.pred_len,
        embedding_dim = args.embedding_dim,
        h_dim = args.encoder_h_dim_d,
        mlp_dim = args.decoder_mlp_dim,
        batch_norm = args.batch_norm,
        dropout = args.dropout,
        num_layers = args.num_layers,
        d_type = args.d_type
    )
    discriminator.apply(init_weights)
    discriminator.type(float_dtype)
    discriminator.train()
    logger.info('The discriminator is demonstrated below: ')
    logger.info(discriminator)
    
    g_loss_fn = g_loss
    d_loss_fn = d_loss
    
    optimizer_g = optim.Adam(generator.parameters(), lr = args.g_learning_rate)
    optimizer_d = optim.Adam(discriminator.parameters(), lr = args.d_learning_rate)
    
    
    # Maybe restore from checkpoint
    restore_path = None
    if args.checkpoint_start_from is not None:
        restore_path = args.checkpoint_start_from
    elif args.restore_from_checkpoint == 1:
        restore_path = os.path.join(args.output_dir,
                                    '%s_with_model.pt' % args.checkpoint_name)

    if restore_path is not None and os.path.isfile(restore_path):
        logger.info('Restoring from checkpoint %s' % restore_path)
        checkpoint = torch.load(restore_path)
        generator.load_state_dict(checkpoint['g_state'])
        discriminator.load_state_dict(checkpoint['d_state'])
        optimizer_g.load_state_dict(checkpoint['g_optim_state'])
        optimizer_d.load_state_dict(checkpoint['d_optim_state'])
        t = checkpoint['counters']['t']
        epoch = checkpoint['counters']['epoch']
        checkpoint['restore_ts'].append(t)
    else:
        # Starting from scratch, so initialize checkpoint data structure
        t, epoch = 0, 0
        checkpoint = {
            'args': args.__dict__,
            'G_losses': defaultdict(list),
            'D_losses': defaultdict(list),
            'losses_ts': [],
            'metrics_val': defaultdict(list),
            'metrics_train': defaultdict(list),
            'sample_ts': [],
            'restore_ts': [],
            'norm_g': [],
            'norm_d': [],
            'counters': {
                't': None,
                'epoch': None,
            },
            'g_state': None,
            'g_optim_state': None,
            'g_best_state': None,
            'd_state': None,
            'd_optim_state': None,
            'd_best_state': None,
            'best_t': None,
            'g_best_nl_state': None,
            'd_best_nl_state': None,
            'best_t_nl': None,
        }
        
    
    t0 = None # The very start of the training process
    while t < args.num_iterations:
        g_steps_left = args.g_steps
        d_steps_left = args.d_steps
        
        epoch += 1
        logger.info('Starting epoch %d' % epoch)
        for batch in train_loader:
            if args.timing == 1:
                torch.cuda.synchronize()
                t1 = time.time()
                
            if d_steps_left > 0:
                step_type = 'd'
                losses_d = discriminator_step(args, batch, generator, discriminator, d_loss_fn, optimizer_d)
                checkpoint['norm_d'].append(get_total_norm(discriminator.parameters()))
                d_steps_left -= 1
                
            elif g_steps_left > 0:
                step_type = 'g'
                #print(next(generator.parameters()).device)
                losses_g = generator_step(args, batch, generator, discriminator, g_loss_fn, optimizer_g)
                checkpoint['norm_g'].append(get_total_norm(generator.parameters()))
                g_steps_left -= 1
            
            
            if args.timing == 1:
                torch.cuda.synchronize()
                t2 = time.time()
                logger.info('%s step took %.2f.' % (step_type, t2 - t1))
            
            if g_steps_left > 0 or d_steps_left > 0:
                continue
            
            
            if args.timing == 1:
                if t0 is not None:
                    logger.info('Iteration %d took %.2f' % (t - 1, time.time() - t0))
                
                t0 = time.time()
            
            
            # Maybe save loss
            if t % args.print_every == 0:
                logger.info('t = %d / %d' % (t + 1, args.num_iterations))
                for k, v in sorted(losses_d.items()):
                    logger.info('[D] %s: %.3f' % (k, v))
                    checkpoint['D_losses'][k].append(v)
                
                for k, v in sorted(losses_g.items()):
                    logger.info('[G] %s: %.3f' % (k, v))
                    checkpoint['G_losses'][k].append(v)
                
                checkpoint['losses_ts'].append(t)
                
            # Maybe save a checkpoint
            if t > 0 and t % args.checkpoint_every == 0:
                checkpoint['counters']['t'] = t
                checkpoint['counters']['epoch'] = epoch
                checkpoint['sample_ts'].append(t)
                
                # Check on the validation set
                logger.info('Checking on validation set...')
                metrics_val = check_accuracy(args, val_loader, generator, discriminator, d_loss_fn)
                
                #Check on the training set
                logger.info('Checking on training set...')
                metrics_train = check_accuracy(args, train_loader, generator, discriminator, d_loss_fn, limit = True)
                
                # Save to the checkpoint file
                for k, v in sorted(metrics_val.items()):
                    logger.info('[val] %s: %.3f' % (k, v))
                    checkpoint['metrics_val'][k].append(v)
                for k, v in sorted(metrics_train.items()):
                    logger.info('[train] %s: %.3f' % (k, v))
                    checkpoint['metrics_train'][k].append(v)
                
                min_ade = min(checkpoint['metrics_val']['ade'])
                min_ade_nl = min(checkpoint['metrics_val']['ade_nl'])
                
                if metrics_val['ade'] == min_ade:
                    logger.info('New low for avg_disp_error')
                    checkpoint['best_t'] = t
                    checkpoint['g_best_state'] = generator.state_dict()
                    checkpoint['d_best_state'] = discriminator.state_dict()

                if metrics_val['ade_nl'] == min_ade_nl:
                    logger.info('New low for avg_disp_error_nl')
                    checkpoint['best_t_nl'] = t
                    checkpoint['g_best_nl_state'] = generator.state_dict()
                    checkpoint['d_best_nl_state'] = discriminator.state_dict()
                    
                # Save another checkpoint with model weights and
                # optimizer state
                checkpoint['g_state'] = generator.state_dict()
                checkpoint['g_optim_state'] = optimizer_g.state_dict()
                checkpoint['d_state'] = discriminator.state_dict()
                checkpoint['d_optim_state'] = optimizer_d.state_dict()
                checkpoint_path = os.path.join(
                    args.output_dir, '%s_with_model.pt' % args.checkpoint_name
                )
                logger.info('Saving checkpoint to %s' % checkpoint_path)
                torch.save(checkpoint, checkpoint_path)
                logger.info('Done')
                
                # Save a checkpoint with no model weights by making a shallow
                # copy of the checkpoint excluding some items
                checkpoint_path = os.path.join(
                    args.output_dir, '%s_no_model.pt' % args.checkpoint_name)
                logger.info('Saving checkpoint to %s' % checkpoint_path)
                key_blacklist = [
                    'g_state', 'g_best_state', 'g_best_nl_state',
                    'g_optim_state', 'd_state', 'd_best_state',
                    'd_best_nl_state', 'd_optim_state']
                small_checkpoint = {}
                for k, v in checkpoint.items():
                    if k not in key_blacklist:
                        small_checkpoint[k] = v
                torch.save(small_checkpoint, checkpoint_path)
                logger.info('Done')
                
            
            t += 1
            g_steps_left = args.g_steps
            d_steps_left = args.d_steps
            
            if t >= args.num_iterations:
                break
            
            
def discriminator_step(args, batch, generator, discriminator, d_loss_fn, optimizer_d):
    batch = [elem.cuda() for elem in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_rel_gt, non_linear_ped, loss_mask, seq_start_end) = batch
    
    loss = torch.zeros(1).to(pred_traj_gt)
    losses = {}
    
    pred_traj_fake, pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end)
    
    traj_gt = torch.cat([obs_traj, pred_traj_gt], dim = 0)
    traj_gt_rel = torch.cat([obs_traj_rel, pred_traj_rel_gt], dim = 0)
    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim = 0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim = 0)
    
    scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
    scores_real = discriminator(traj_gt, traj_gt_rel, seq_start_end)
    
    discriminator_loss = d_loss_fn(scores_real, scores_fake)
    losses['D_data_loss'] = discriminator_loss.item()
    loss += discriminator_loss
    losses['D_total_loss'] = loss.item()
    
    optimizer_d.zero_grad()
    loss.backward()
    
    if args.clip_threshold_d > 0:
        nn.utils.clip_grad_norm_(discriminator.parameters(), args.clip_threshold_d)
    
    optimizer_d.step()
    
    return losses
                

def generator_step(args, batch, generator, discriminator, g_loss_fn, optimizer_g):
    #print(next(generator.parameters()).device)
    batch = [elem.cuda() for elem in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_rel_gt, non_linear_ped, loss_mask, seq_start_end) = batch
    
    loss = torch.zeros(1).to(pred_traj_gt)
    losses = {}
    g_l2_loss_rel = []
    
    loss_mask = loss_mask[:, args.obs_len:]
    
    for _ in range(args.best_k):
        pred_traj_fake, pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end)
        
        if args.l2_loss_weight > 0:
            g_l2_loss_rel.append(args.l2_loss_weight * l2_loss(pred_traj_fake_rel, pred_traj_rel_gt,
                                                               loss_mask, mode = 'raw'))
    
    g_l2_loss_rel_sum = torch.zeros(1).to(pred_traj_gt)
    if args.l2_loss_weight > 0:
        g_l2_loss_rel = torch.stack(g_l2_loss_rel, dim = 1)
        for _, (start, end) in enumerate(seq_start_end):
            _g_l2_loss_rel = g_l2_loss_rel[start: end]
            _g_l2_loss_rel = torch.sum(_g_l2_loss_rel, dim = 0)
            _g_l2_loss_rel = torch.min(_g_l2_loss_rel) / torch.sum(loss_mask[start: end])
            g_l2_loss_rel_sum += _g_l2_loss_rel
        losses['G_l2_loss_rel'] = g_l2_loss_rel_sum.item()
        loss += g_l2_loss_rel_sum
        
    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim = 0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim = 0)
    scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
    discriminator_loss = g_loss_fn(scores_fake)
    
    loss += discriminator_loss
    losses['G_discriminator_loss'] = discriminator_loss.item()
    losses['G_total_loss'] = loss.item()
    
    optimizer_g.zero_grad()
    loss.backward()
    if args.clip_threshold_g > 0:
        nn.utils.clip_grad_norm_(generator.parameters(), max_norm = args.clip_threshold_g)
    optimizer_g.step()
    
    return losses



def check_accuracy(args, loader, generator, discriminator, d_loss_fn, limit = False):
    metrics = {}
    d_losses = []
    g_l2_losses_abs, g_l2_losses_rel = ([], ) * 2
    disp_error, disp_error_l, disp_error_nl = ([], ) * 3
    f_disp_error, f_disp_error_l, f_disp_error_nl = ([], ) * 3
    total_traj, total_traj_l, total_traj_nl = 0, 0, 0
    loss_mask_sum = 0
    generator.eval()
    
    with torch.no_grad():
        for batch in loader:
            batch = [elem.cuda() for elem in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, 
             non_linear_ped, loss_mask, seq_start_end) = batch
            linear_ped = 1 - non_linear_ped
            loss_mask = loss_mask[:, args.obs_len:]
            
            pred_traj_fake, pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end)
            
            g_l2_loss_abs, g_l2_loss_rel = cal_l2_losses(pred_traj_gt, pred_traj_gt_rel, pred_traj_fake,
                pred_traj_fake_rel, loss_mask)
            
            ade, ade_l, ade_nl = cal_ade(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped)
            
            fde, fde_l, fde_nl = cal_fde(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped)
            
            traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
            traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
            traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
            traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)
            
            scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
            scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)
            
            d_loss = d_loss_fn(scores_real, scores_fake)
            d_losses.append(d_loss.item())
            
            g_l2_losses_abs.append(g_l2_loss_abs.item())
            g_l2_losses_rel.append(g_l2_loss_rel.item())
            disp_error.append(ade.item())
            disp_error_l.append(ade_l.item())
            disp_error_nl.append(ade_nl.item())
            f_disp_error.append(fde.item())
            f_disp_error_l.append(fde_l.item())
            f_disp_error_nl.append(fde_nl.item())
            
            loss_mask_sum += torch.numel(loss_mask.data)
            total_traj += pred_traj_gt.size(1)
            total_traj_l += torch.sum(linear_ped).item()
            total_traj_nl += torch.sum(non_linear_ped).item()
            
            if limit and total_traj >= args.num_samples_check:
                break
        
    metrics['d_loss'] = sum(d_losses) / len(d_losses)
    metrics['g_l2_loss_abs'] = sum(g_l2_losses_abs) / loss_mask_sum
    metrics['g_l2_loss_rel'] = sum(g_l2_losses_rel) / loss_mask_sum

    metrics['ade'] = sum(disp_error) / (total_traj * args.pred_len)
    metrics['fde'] = sum(f_disp_error) / total_traj
    if total_traj_l != 0:
        metrics['ade_l'] = sum(disp_error_l) / (total_traj_l * args.pred_len)
        metrics['fde_l'] = sum(f_disp_error_l) / total_traj_l
    else:
        metrics['ade_l'] = 0
        metrics['fde_l'] = 0
    if total_traj_nl != 0:
        metrics['ade_nl'] = sum(disp_error_nl) / (
            total_traj_nl * args.pred_len)
        metrics['fde_nl'] = sum(f_disp_error_nl) / total_traj_nl
    else:
        metrics['ade_nl'] = 0
        metrics['fde_nl'] = 0

    generator.train()
    return metrics
            
            
            


def cal_l2_losses(pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, pred_traj_fake_rel, loss_mask):
    g_l2_loss_abs = l2_loss(pred_traj_fake, pred_traj_gt, loss_mask, mode = 'sum')
    g_l2_loss_rel = l2_loss(pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode = 'sum')
    return g_l2_loss_abs, g_l2_loss_rel


def cal_ade(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped):
    ade = displacement_error(pred_traj_fake, pred_traj_gt)
    ade_l = displacement_error(pred_traj_fake, pred_traj_gt, linear_ped)
    ade_nl = displacement_error(pred_traj_fake, pred_traj_gt, non_linear_ped)
    return ade, ade_l, ade_nl


def cal_fde(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped):
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
    fde_l = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], linear_ped)
    fde_nl = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], non_linear_ped)
    return fde, fde_l, fde_nl


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)