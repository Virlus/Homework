import argparse
import os
import torch
import sys

from attrdict import AttrDict

curr_path = os.path.abspath(__file__)
curr_path = curr_path.split('/')[:-2]
curr_path = '/'.join(curr_path)
sys.path.append(curr_path)

from slstm.data.dataloader import data_loader
from slstm.models import TrajectoryGenerator
from slstm.losses import displacement_error, final_displacement_error
from slstm.utils import get_dataset_path

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)


def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        input_dim = 2, 
        embedding_dim = args.embedding_dim, 
        encoder_h_dim = args.encoder_h_dim_g, 
        decoder_h_dim = args.decoder_h_dim_g, 
        num_layers = args.num_layers, 
        batch_norm = args.batch_norm, 
        obs_len = args.obs_len, 
        pred_len = args.pred_len, 
        dropout = args.dropout,
        activation = 'relu', 
        neighbourhood_size = args.neighbourhood_size, 
        grid_size = args.neighbourhood_size, 
        pool_h_dim = args.pool_h_dim
    )
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    
    return generator


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
            
            pred_traj_fake, pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end)
            ade.append(displacement_error(pred_traj_fake, pred_traj_gt, mode = 'sum'))
            fde.append(final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], mode = 'sum'))
                
            
            ade_outer.append(ade[0])
            fde_outer.append(fde[0])
            
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
