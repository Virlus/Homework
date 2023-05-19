import torch
import random


def l2_loss(pred_traj, pred_traj_gt, loss_mask, mode = 'sum'):
    '''
    Input:
        pred_traj: tensor of shape (pred_len, batch, 2)
        pred_traj_gt: tensor of shape (pred_len, batch, 2)
        loss_mask: tensor of shape (batch, pred_len)
        mode: 'average' or 'sum' or 'raw'
    Output:
        l2 loss depending on the current mode
     '''
     
    pred_len, batch, _ = pred_traj.size()
    l2_losses = loss_mask.unsqueeze(dim = 2) * (pred_traj.permute(1, 0, 2) - pred_traj_gt.permute(1, 0, 2)) ** 2
    
    if mode == 'sum':
        return torch.sum(l2_losses)
    elif mode == 'average':
        return torch.sum(l2_losses) / torch.numel(l2_losses.data)
    elif mode == 'raw':
        return l2_losses.sum(dim = 2).sum(dim = 1)


def displacement_error(pred_traj, pred_traj_gt, to_consider = None, mode = 'sum'):
    '''
    Input:
        pred_traj: tensor of shape (pred_len, batch, 2)
        pred_traj_gt: tensor of shape (pred_len, batch, 2)
        to_consider: tensor of shape (batch, ). Might not be given
        mode: 'sum' or 'raw'
    Output:
        disp_error: dependent on current mode
    '''
    disp_error = pred_traj.permute(1, 0, 2) - pred_traj_gt.permute(1, 0, 2)
    disp_error = disp_error ** 2
    if to_consider is not None:
        disp_error = torch.sqrt(disp_error.sum(dim = 2)).sum(dim = 1) * to_consider
    else:
        disp_error = torch.sqrt(disp_error.sum(dim = 2)).sum(dim = 1)
    
    if mode == 'sum':
        disp_error = torch.sum(disp_error)
        return disp_error
    elif mode == 'raw':
        return disp_error
    
    
    
def final_displacement_error(pred_pos, pred_pos_gt, to_consider = None, mode = 'sum'):
    '''
    Input:
        pred_pos: tensor of shape (batch, 2)
        pred_pos_gt: tensor of shape (batch, 2)
        to_consider: tensor of shape (batch, ). Might not be given
        mode: 'sum' or 'raw'
    Output:
        disp_error: dependent on current mode
    '''
    disp_error = (pred_pos - pred_pos_gt) ** 2
    if to_consider is not None:
        disp_error = torch.sqrt(disp_error.sum(dim = 1)) * to_consider
    else:
        disp_error = torch.sqrt(disp_error.sum(dim = 1))
    
    if mode == 'sum':
        return torch.sum(disp_error)
    else:
        return disp_error