import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import sys
import math
import logging


logger = logging.getLogger(__name__)



def seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list, 
     non_linear_ped_list, loss_mask_list) = zip(*data)
    
    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end] for start, end in zip(cum_start_idx[:-1], cum_start_idx[1:])]
    
    obs_traj = torch.cat(obs_seq_list, dim = 0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim = 0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim = 0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim = 0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim = 0)
    seq_start_end = torch.LongTensor(seq_start_end)
    # Output format: (seq_len, batch, input_size)
    output = [obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped, loss_mask, seq_start_end]
    
    return tuple(output)
    



def read_file(_path, delim = 'tab'):
    data = []
    if delim == 'tab':
        delim = '\t'
    with open(_path, 'r') as file:
        for line in file:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    
    data = np.asarray(data)
    return data


def polyjudge(traj, traj_len, threshold):
    '''
    Input:
        traj: ndarray of shape (2, traj_len)
        traj_len: length of trajectory
        threshold: The lower bound of non-linearity
    
    Output:
        1: non-linear
        0: linear
    '''
    timestamps = np.linspace(0, traj_len - 1, traj_len)
    loss_x = np.polyfit(timestamps, traj[0, :], deg = 2, full = True)[1]
    loss_y = np.polyfit(timestamps, traj[1, :], deg = 2, full = True)[1]
    
    if loss_x + loss_y >= threshold:
        return 1.0
    else:
        return 0.0
    



class TrajectoriesDataset(Dataset):
    '''
    Class generating a dataset from the given directory.
    '''
    
    def __init__(self, data_dir = None, threshold = 0.002, delim = 'tab', 
                 obs_len = 8, pred_len = 12, min_ped = 1, skip = 1):
        '''
        Args:
            data_dir: Directory containing all the files.
            threshold: The lower bound of non-linear judgements.
            delim: The delimiter in raw files.
            obs_len: The length of observed sequence.
            pred_len: The length of predicted sequence.
            min_pred: The least possible number of pedestrians in a frame.
        '''
        
        super(TrajectoriesDataset, self).__init__()
        
        self.seq_len = obs_len + pred_len
        self.threshold = threshold
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.data_dir = data_dir
        self.min_ped = min_ped
        self.delim = delim
        self.skip = skip
        
        if data_dir is None:
            logger.info("No directory specified")
        
        all_files = os.listdir(data_dir)
        all_files = [os.path.join(data_dir, file) for file in all_files]
        
        num_peds_in_seq = [] # (# of frames, )
        seq_list = [] # (# of frames, # of pedestrians, 2, seq_len)
        seq_list_rel = [] # (# of frames, # of pedestrians, 2, seq_len)
        loss_mask_list = [] # (# of frames, # of pedestrians, seq_len)
        non_linear_list = [] # (# of frames, # of pedestrians)
        
        for _path in all_files:
            data = read_file(_path, self.delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            
            for frame in frames:
                frame_data.append(data[data[:, 0] == frame, :])
                
            num_seqs = int(math.ceil((len(frames) - self.seq_len + 1) / self.skip))
            
            for idx in range(0, self.skip * num_seqs + 1, self.skip):
                potential_seq = frame_data[idx: idx + self.seq_len]
                potential_seq = np.concatenate(potential_seq, axis = 0)
                occurring_peds = np.unique(potential_seq[:, 1]).tolist()
                max_ped_num = len(occurring_peds)
                peds_seq = np.zeros((max_ped_num, 2, self.seq_len))
                peds_seq_rel = np.zeros((max_ped_num, 2, self.seq_len))
                loss_mask = np.zeros((max_ped_num, self.seq_len))
                non_linear = []
                valid_peds = 0
                
                for _, occurring_ped in enumerate(occurring_peds):
                    ped_seq = potential_seq[potential_seq[:, 1] == occurring_ped, :]
                    ped_seq = np.around(ped_seq, decimals = 4)
                    ped_start = frames.index(ped_seq[0, 0])
                    ped_end = frames.index(ped_seq[-1, 0])
                    #print(ped_start, ped_end)
                    
                    if ped_end - ped_start != self.seq_len - 1: # this particular pedestrian doesn't match the sequence
                        continue
                    
                    ped_pos = ped_seq[:, 2:]
                    ped_pos = np.transpose(ped_pos)
                    ped_pos_rel = np.zeros_like(ped_pos)
                    ped_pos_rel[:, 1:] = ped_pos[:, 1:] - ped_pos[:, :-1]
                    peds_seq[valid_peds, :, 0: self.seq_len] = ped_pos
                    peds_seq_rel[valid_peds, :, 0: self.seq_len] = ped_pos_rel
                    loss_mask[valid_peds, :] = 1
                    non_linear.append(polyjudge(ped_pos, self.seq_len, self.threshold))
                    
                    valid_peds += 1
                
                #print(valid_peds) # always 0
                if valid_peds >= self.min_ped:
                    seq_list.append(peds_seq[:valid_peds])
                    seq_list_rel.append(peds_seq_rel[:valid_peds])
                    loss_mask_list.append(loss_mask[:valid_peds])
                    num_peds_in_seq.append(valid_peds)
                    non_linear_list += non_linear
                    
        
        self.num_seq = len(num_peds_in_seq)
        seq_list = np.concatenate(seq_list, axis = 0)
        seq_list_rel = np.concatenate(seq_list_rel, axis = 0)
        loss_mask_list = np.concatenate(loss_mask_list, axis = 0)
        non_linear_list = np.asarray(non_linear_list)
        
        self.obs_traj = torch.from_numpy(seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_list).type(torch.float)
        
        cumsum_peds_in_seq = np.cumsum(num_peds_in_seq).tolist()
        cumsum_peds_in_seq = [0] + cumsum_peds_in_seq
        
        self.seq_start_end = [(start, end) for start, end in zip(cumsum_peds_in_seq[:-1], cumsum_peds_in_seq[1:])]
        
        
    def __len__(self):
        return self.num_seq
    
    
    def __getitem__(self, idx):
        start, end = self.seq_start_end[idx]
        
        output = [self.obs_traj[start: end, :], self.pred_traj[start: end, :], 
                  self.obs_traj_rel[start: end, :], self.pred_traj_rel[start: end, :],
                  self.non_linear_ped[start: end], self.loss_mask[start: end, :]]
        
        return output
    
        
        
        
        
                    
                    
            
                
            
        
        
        
        
        