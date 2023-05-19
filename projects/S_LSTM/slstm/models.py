import torch
import torch.nn as nn



def mlp(dim_list, batch_norm = 1, dropout = 0.0, activation = 'relu'):
    nets = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        nets.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            nets.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            nets.append(nn.ReLU())
        elif activation == 'leakyrelu':
            nets.append(nn.LeakyReLU())
        if dropout > 0:
            nets.append(nn.Dropout(dropout))
    
    nets = nn.Sequential(*nets)
    
    return nets


class Encoder(nn.Module):
    '''
    Encode the observed sequence using only the naive LSTM.
    '''
    def __init__(self, input_dim = 2, embedding_dim = 64, hidden_dim = 64, num_layers = 1, 
                 batch_norm = 1, obs_len = 8, dropout = 0.0):
        
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.obs_len = obs_len
        self.dropout = dropout
        
        
        self.embedding = nn.Linear(input_dim, embedding_dim)
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = hidden_dim, 
                            num_layers = num_layers, dropout = dropout)
        
        
    def get_hidden(self, batch):
        return (torch.zeros(self.num_layers, batch, self.hidden_dim).cuda(), 
                torch.zeros(self.num_layers, batch, self.hidden_dim).cuda())
        
    
    def forward(self, obs_seq):
        '''
        Input:
            obs_seq: tensor of shape (self.obs_len, batch, self.input_dim)
        Output:
            state: tuple with two tensors of shape (self.num_layers, batch, self.hidden_dim)
        '''
        batch = obs_seq.shape[1]
        obs_seq = obs_seq.reshape(-1, self.input_dim)
        obs_seq = self.embedding(obs_seq)
        obs_seq = obs_seq.reshape(-1, batch, self.embedding_dim)
        
        _, state = self.lstm(obs_seq, self.get_hidden(batch))
        
        return state
    
    

class Decoder(nn.Module):
    '''
    Predict the future trajectories using naive lstm.
    '''
    def __init__(self, input_dim = 2, embedding_dim = 2, hidden_dim = 64, num_layers = 1,
                 batch_norm = 1, pred_len = 12, dropout = 0.0):
        
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_len = pred_len
        self.dropout = dropout
        
        self.embedding = nn.Linear(input_dim, embedding_dim)
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = hidden_dim, num_layers = num_layers,
                            dropout = dropout)
        self.hidden2pos = nn.Linear(hidden_dim, 2)
        
        
    def forward(self, encoder_state, obs_final_pos, obs_final_pos_rel):
        '''
        Input:
            encoder_state: tuple of two tensors of shape (1, batch, self.hidden_dim)
            obs_final_pos: tensor of shape (batch, 2)
            obs_final_pos_rel: tensor of shape (batch, 2)
        
        Output:
            pred_seq: tensor of shape (self.pred_len, batch, 2)
            pred_seq_rel: tensor of shape (self.pred_len, batch, 2) 
        '''
        encoder_final_h = encoder_state[0]
        encoder_final_c = encoder_state[1]
        state_tuple = (encoder_final_h, encoder_final_c)
        
        decoder_input = self.embedding(obs_final_pos_rel)
        decoder_input = decoder_input.unsqueeze(0)
        last_pos = obs_final_pos
        
        pred_seq = []
        pred_seq_rel = []
        
        for _ in range(self.pred_len):
            output, state_tuple = self.lstm(decoder_input, state_tuple)
            output = output.reshape(-1, self.hidden_dim)
            rel_pos = self.hidden2pos(output)
            curr_pos = rel_pos + last_pos
            pred_seq.append(curr_pos)
            pred_seq_rel.append(rel_pos)
            last_pos = curr_pos
            
            decoder_input = self.embedding(rel_pos)
            decoder_input = decoder_input.unsqueeze(0)
        
        pred_seq = torch.stack(pred_seq, dim = 0)
        pred_seq_rel = torch.stack(pred_seq_rel, dim = 0)
        
        return pred_seq, pred_seq_rel
    
    
class SocialPooling(nn.Module):
    '''
    Pooling Module meant for capturing adjacent information.
    '''
    def __init__(self, pool_h_dim = 64, batch_norm = 1, dropout = 0.0, activation = 'relu',
                 neighbourhood_size = 2.0, grid_size = 8, pool_out_dim = None):
        super(SocialPooling, self).__init__()
        
        self.pool_h_dim = pool_h_dim
        self.neighbourhood_size = neighbourhood_size
        self.grid_size = grid_size
        self.dropout = dropout
        if pool_out_dim is not None:
            self.pool_out_dim = pool_out_dim
        else:
            self.pool_out_dim = pool_h_dim
            
        mlp_in_dim = int(grid_size * grid_size * pool_h_dim)
        mlp_dim_list = [mlp_in_dim, self.pool_out_dim]
        self.mlp = mlp(mlp_dim_list, batch_norm = batch_norm, dropout = dropout, activation = activation)
        
    
    def find_bbox(self, curr_pos):
        top_left_x = curr_pos[:, 0] - self.neighbourhood_size / 2
        top_left_y = curr_pos[:, 1] + self.neighbourhood_size / 2
        bottom_right_x = curr_pos[:, 0] + self.neighbourhood_size / 2
        bottom_right_y = curr_pos[:, 1] - self.neighbourhood_size / 2
        
        top_left = torch.stack([top_left_x, top_left_y], dim = 1)
        bottom_right = torch.stack([bottom_right_x, bottom_right_y], dim = 1)
        
        return top_left, bottom_right
    
    
    def find_grid(self, curr_pos, top_left):
        grid_idx_x = torch.floor((curr_pos[:, 0] - top_left[:, 0]) / self.neighbourhood_size * self.grid_size)
        grid_idx_y = torch.floor((top_left[:, 1] - curr_pos[:, 1]) / self.neighbourhood_size * self.grid_size)
        grid_idx = grid_idx_x + self.grid_size * grid_idx_y # shape (batch, )
        return grid_idx
    
        
    def forward(self, hidden_states, all_pos, seq_start_end):
        '''
        Input:
            hidden_states: tensor of shape (batch, lstm_h_dim)
            all_pos: tensor of shape (batch, 2)
            seq_start_end: tensor delimiting the start and end of a certain frame
        
        Output:
            pool_output: tensor of shape (batch, self.pool_out_dim)
        '''
        
        #hidden_states = hidden_states[0] # Default # of layers = 1
        pool_h = [] # The list of the pooling hidden states of all frames
        
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_peds = end - start
            
            total_grid_num = int(num_peds * self.grid_size * self.grid_size + 1)
            # total_grid_num = torch.tensor(total_grid_num)
            # pool_h_dim_trans = torch.tensor(self.pool_h_dim)
            curr_pool_h = torch.zeros(total_grid_num, self.pool_h_dim).type_as(hidden_states) # The final tensor to append
            curr_h_states = hidden_states[start: end] # The collections of hidden states in the current frame
            curr_h_states_repeat = curr_h_states.repeat(num_peds, 1) # Shape (num_peds * num_peds, self.pool_h_dim)
            
            curr_pos = all_pos[start:end]
            curr_pos_repeat = curr_pos.repeat(num_peds, 1) # P1, P2, P3, P1, P2, P3, ...
            curr_pos_interleave = torch.repeat_interleave(curr_pos, num_peds, dim = 0)
            
            top_left, bottom_right = self.find_bbox(curr_pos_interleave) # B1, B1, B1, B2, B2, B2, B3, B3, B3,...
            grid_idx = self.find_grid(curr_pos_repeat, top_left).type_as(seq_start_end) # Shape (num_ped ** 2, )
            grid_idx += 1 # In order to save a row for waste information in curr_pool_h
            
            mask = (curr_pos_repeat[:, 0] <= top_left[:, 0]) + (curr_pos_repeat[:, 1] >= top_left[:, 1]) + (curr_pos_repeat[:, 0] >= bottom_right[:, 0]) + (curr_pos_repeat[:, 1] <= bottom_right[:, 1]) # Shape (num_ped ** 2, )
            mask[0: : num_peds + 1] = 1 # One shouldn't be considered in its own neighbourhood
            mask = mask.reshape(-1)
            
            total_grid_size = self.grid_size * self.grid_size
            offset = torch.arange(0, total_grid_size * num_peds, total_grid_size).type_as(seq_start_end)
            offset = offset.repeat_interleave(num_peds)
            grid_idx += offset
            
            grid_idx[mask != 0] = 0 # Clear the grid index outside the bounding boxes
            grid_idx = grid_idx.reshape(-1, 1) # Reshaped as (num_peds * num_peds, self.pool_h_dim)
            grid_idx = grid_idx.repeat(1, curr_h_states.size(1))

            curr_pool_h = curr_pool_h.scatter_add(0, grid_idx, curr_h_states_repeat)
            curr_pool_h = curr_pool_h[1:]
            
            pool_h.append(curr_pool_h.reshape(num_peds, -1))
            
        pool_h = torch.cat(pool_h, dim = 0)
        pool_output = self.mlp(pool_h)
        return pool_output
    
            
            
class TrajectoryGenerator(nn.Module):
    '''
    The combination of encoder and decoder, exploited to predict the future trajectories.
    '''
    def __init__(self, input_dim = 2, embedding_dim = 64, encoder_h_dim = 128, decoder_h_dim = 128, 
                 num_layers = 1, batch_norm = 1, obs_len = 8, pred_len = 12, dropout = 0.0,
                 activation = 'relu', neighbourhood_size = 2.0, grid_size = 8, pool_h_dim = 128):
        
        super(TrajectoryGenerator, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.num_layers = num_layers
        self.obs_len = obs_len
        self.pred_len = pred_len
        #self.dropout = dropout
        

        self.embedding = nn.Linear(input_dim, embedding_dim)
        self.hidden2pos = nn.Linear(decoder_h_dim, 2)
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = decoder_h_dim, num_layers = num_layers,
                            dropout = dropout)
        self.pool_net = SocialPooling(pool_h_dim = pool_h_dim, batch_norm = batch_norm, dropout = dropout,
                                      activation = activation, neighbourhood_size = neighbourhood_size, 
                                      grid_size = grid_size, pool_out_dim = None)
        
        hidden_mlp_list = [decoder_h_dim + pool_h_dim, decoder_h_dim]
        self.hidden_mlp = mlp(hidden_mlp_list, batch_norm = batch_norm, dropout = dropout, activation = 'relu')
        
    
    def get_hidden(self, batch):
        return (torch.zeros(self.num_layers, batch, self.decoder_h_dim).cuda(), 
                torch.zeros(self.num_layers, batch, self.decoder_h_dim).cuda())
        
        
    def forward(self, obs_seq, obs_seq_rel, seq_start_end):
        '''
        Input:
            obs_seq: tensor of shape (obs_len, batch, 2)
            obs_seq_rel: tensor of shape (obs_len, batch, 2)
        
        Output:
            pred_seq: tensor of shape (pred_len, batch, 2)
            pred_seq_rel: tensor of shape (pred_len, batch, 2)
        '''
        batch = obs_seq_rel.size(1)
        obs_seq_rel_embed = obs_seq_rel.reshape(-1, self.input_dim)
        obs_seq_rel_embed = self.embedding(obs_seq_rel_embed)
        obs_seq_rel_embed = obs_seq_rel_embed.reshape(-1, batch, self.embedding_dim)
        
        _, state = self.lstm(obs_seq_rel_embed, self.get_hidden(batch))
        
        encoder_last_pos = obs_seq[-1]
        encoder_last_pos_rel = obs_seq_rel[-1]
        
        state_tuple = state
        
        pred_seq = []
        pred_seq_rel = []
        
        decoder_input = self.embedding(encoder_last_pos_rel)
        decoder_input = decoder_input.unsqueeze(0)
        last_pos = encoder_last_pos
        
        #pool_h = self.pool_net(state_tuple[0][0], last_pos, seq_start_end)
        
        
        for _ in range(self.pred_len):
            output, state_tuple = self.lstm(decoder_input, state_tuple)
            output = output.reshape(-1, self.decoder_h_dim)
            rel_pos = self.hidden2pos(output)
            curr_pos = rel_pos + last_pos
            pred_seq.append(curr_pos)
            pred_seq_rel.append(rel_pos)
            last_pos = curr_pos
            
            pool_h = self.pool_net(state_tuple[0][0], last_pos, seq_start_end)
            mlp_input = torch.cat([state_tuple[0][0], pool_h], dim = 1)
            new_hidden_state = self.hidden_mlp(mlp_input)
            new_hidden_state = new_hidden_state.unsqueeze(0)
            state_tuple = (new_hidden_state, state_tuple[1])
            
            decoder_input = self.embedding(rel_pos)
            decoder_input = decoder_input.unsqueeze(0)
        
        pred_seq = torch.stack(pred_seq, dim = 0)
        pred_seq_rel = torch.stack(pred_seq_rel, dim = 0)
        
        return pred_seq, pred_seq_rel
    

        