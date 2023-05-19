import argparse
import os
import torch
import sys

from attrdict import AttrDict

curr_path = os.path.abspath(__file__)
curr_path = curr_path.split('/')[:-2]
curr_path = '/'.join(curr_path)
sys.path.append(curr_path)

from sgan.data.dataloader import data_loader
from sgan.models import TrajectoryGenerator, TrajectoryDiscriminator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import get_dataset_path

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)


def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
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
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    
    return generator


# def get_discriminator(checkpoint):
#     args = AttrDict(checkpoint['args'])
#     discriminator = TrajectoryDiscriminator(
#         obs_len = args.obs_len,
#         pred_len = args.pred_len,
#         embedding_dim = args.embedding_dim,
#         h_dim = args.encoder_h_dim_d,
#         mlp_dim = args.decoder_mlp_dim,
#         batch_norm = args.batch_norm,
#         dropout = args.dropout,
#         num_layers = args.num_layers,
#         d_type = args.d_type
#     )
#     discriminator.load_state_dict(checkpoint['d_state'])
#     discriminator.cuda()
#     discriminator.train()
    
#     return discriminator

def evaluate_helper(error, seq_start_end):
    _sum = 0
    error = torch.stack(error, dim = 1)
    
    for _, (start, end) in enumerate(seq_start_end):
        start = start.item()
        end = end.item()
        _error = error[start: end]
        _error = torch.sum(_error, dim = 0)
        _error = torch.min(_error)
        _sum += _error
    
    return _sum
        


def evaluate(args, loader, generator, num_samples):
    ade_outer, fde_outer = [], []
    total_traj = 0
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch
            
            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)
            
            for _ in range(num_samples):
                pred_traj_fake, pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end)
                ade.append(displacement_error(pred_traj_fake, pred_traj_gt, mode = 'raw'))
                fde.append(final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], mode = 'raw'))
            
            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)
                
            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)
            
        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / (total_traj)
        
        return ade, fde


def main(args):
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [os.path.join(args.model_path, _file) for _file in filenames]
    else:
        paths = [args.model_path]
        
    for path in paths:
        checkpoint = torch.load(path)
        generator = get_generator(checkpoint)
        _args = AttrDict(checkpoint['args'])
        path = get_dataset_path(_args.dataset_name, args.dset_type)
        _, loader = data_loader(_args, path)
        ade, fde = evaluate(_args, loader, generator, args.num_samples)
        print('Dataset: %s, Pred_len: %d, ADE: %.2f, FDE: %.2f' % (_args.dataset_name, _args.pred_len, ade, fde))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
