import torch
import torch.nn as nn



def mlp(dim_list, batch_norm = True, dropout = 0.0, activation = 'relu'):
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


def generate_noise(noise_shape = None, noise_type = 'gaussian'):
    if noise_type == 'gaussian':
        return torch.randn(*noise_shape).cuda()
    elif noise_type == 'uniform':
        return torch.rand(*noise_shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unknown noise type: %s' % noise_type)


class Encoder(nn.Module):
    '''
    Part of both the trajectory generator and the trajectory discriminator.
    '''
    def __init__(self, embedding_dim = 64, hidden_dim = 64, num_layers = 1, 
                 dropout = 0.0):
        
        super(Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        
        self.embedding = nn.Linear(2, embedding_dim)
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = hidden_dim, 
                            num_layers = num_layers, dropout = dropout)
        
        
    def get_hidden(self, batch):
        return (torch.zeros(self.num_layers, batch, self.hidden_dim).cuda(), 
                torch.zeros(self.num_layers, batch, self.hidden_dim).cuda())
        
    
    def forward(self, obs_seq):
        '''
        Input:
            obs_seq: tensor of shape (self.obs_len, batch, 2)
        Output:
            final_encoder_h: tensor of shape (self.num_layers, batch, self.hidden_dim)
        '''
        batch = obs_seq.size(1)
        obs_seq = obs_seq.reshape(-1, 2)
        obs_seq = self.embedding(obs_seq)
        obs_seq = obs_seq.reshape(-1, batch, self.embedding_dim)
        
        _, state = self.lstm(obs_seq, self.get_hidden(batch))
        final_encoder_h = state[0]
        
        return final_encoder_h
    
    

class Decoder(nn.Module):
    '''
    Part of the trajectory generator.
    '''
    def __init__(self, pred_len = 12, embedding_dim = 64, hidden_dim = 128, bottleneck_dim = 1024, 
                 mlp_dim = 1024, num_layers = 1, batch_norm = True, dropout = 0.0, pool_type = 'pool_net',
                 activation = 'relu', pool_every_step = True, neighbourhood_size = 2.0, grid_size = 8,
                 pool_mlp_dim = 512):
        
        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.pred_len = pred_len
        self.dropout = dropout
        self.pool_mlp_dim = pool_mlp_dim
        self.pool_every_step = pool_every_step
        
        if pool_type == 'pool_net':
            self.pool_net = PoolNet(embedding_dim = embedding_dim, h_dim = hidden_dim,
                                    bottleneck_dim = bottleneck_dim, pool_mlp_dim = pool_mlp_dim,
                                    batch_norm = batch_norm, dropout = dropout, activation = activation)
            
        elif pool_type == 'social_pool':
            self.pool_net = SocialPooling(pool_h_dim = bottleneck_dim,
                                          batch_norm = batch_norm,
                                          dropout = dropout,
                                          activation = activation,
                                          neighbourhood_size = neighbourhood_size,
                                          grid_size = grid_size)
        
        self.embedding = nn.Linear(2, embedding_dim)
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = hidden_dim, num_layers = num_layers,
                            dropout = dropout)
        self.hidden2pos = nn.Linear(hidden_dim, 2)
        
        mlp_dim_list = [hidden_dim + bottleneck_dim, mlp_dim, hidden_dim]
        self.mlp = mlp(mlp_dim_list, batch_norm = batch_norm, dropout = dropout, activation = activation)
        
        
    def forward(self, state_tuple, obs_final_pos, obs_final_pos_rel, seq_start_end):
        '''
        Input:
            state_tuple: tuple of tensors of shape (1, batch, self.hidden_dim)
            obs_final_pos: tensor of shape (batch, 2)
            obs_final_pos_rel: tensor of shape (batch, 2)
            seq_start_end: Delimiter of the starts and ends of frames
        
        Output:
            pred_seq: tensor of shape (self.pred_len, batch, 2)
            pred_seq_rel: tensor of shape (self.pred_len, batch, 2) 
        '''
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
            
            if self.pool_every_step:
                pool_h = self.pool_net(state_tuple[0], curr_pos, seq_start_end)
                curr_h = state_tuple[0].reshape(-1, self.hidden_dim)
                mlp_in = torch.cat([curr_h, pool_h], dim = 1)
                new_h = self.mlp(mlp_in)
                new_h = new_h.unsqueeze(0)
                state_tuple = (new_h, state_tuple[1])
                
            pred_seq.append(curr_pos)
            pred_seq_rel.append(rel_pos)
            last_pos = curr_pos
            
            decoder_input = self.embedding(rel_pos)
            decoder_input = decoder_input.unsqueeze(0)
        
        pred_seq = torch.stack(pred_seq, dim = 0)
        pred_seq_rel = torch.stack(pred_seq_rel, dim = 0)
        
        return pred_seq, pred_seq_rel
    
    
class PoolNet(nn.Module):
    '''
    Pooling module inspired by PointNet, also the default pooling method applied in SGAN.
    '''
    def __init__(self, embedding_dim = 64, h_dim = 128, bottleneck_dim = 1024, pool_mlp_dim = 512, 
                 batch_norm = True, dropout = 0.0, activation = 'relu'):
        super(PoolNet, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.pool_mlp_dim = pool_mlp_dim
        
        mlp_in_dim = embedding_dim + h_dim
        mlp_dims = [mlp_in_dim, pool_mlp_dim, bottleneck_dim]
        self.mlp = mlp(mlp_dims, batch_norm = batch_norm, dropout = dropout, activation = activation)
        
        self.embedding = nn.Linear(2, embedding_dim)
        
    def forward(self, curr_h_states, curr_pos, seq_start_end):
        '''
        Input:
            curr_h_states: tensor of shape (num_layers, batch, self.h_dim)
            curr_pos: tensor of shape (batch, 2)
            seq_start_end : delimiter of the start and end of each frame
            
        Output:
            pool_h: tensor of shape (batch, self.bottleneck_dim)
        '''
        batch = curr_h_states.size(1)
        curr_h_states = curr_h_states[0]
        pool_h = []
         
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_peds = end - start
            frame_h_states = curr_h_states[start: end]
            frame_h_states_repeat = frame_h_states.repeat(num_peds, 1) # H1, H2, H3, H1, H2, H3,...
            frame_pos = curr_pos[start: end]
            frame_pos_repeat = frame_pos.repeat(num_peds, 1) # P1, P2, P3, P1, P2, P3, ...
            frame_pos_interleave = torch.repeat_interleave(frame_pos, num_peds, dim = 0) # P1, P1, P1, P2, P2,...
            rel_pos = frame_pos_repeat - frame_pos_interleave
            rel_pos_embedding = self.embedding(rel_pos)
            mlp_input = torch.cat([rel_pos_embedding, frame_h_states_repeat], dim = 1)
            mlp_output = self.mlp(mlp_input)
            frame_pool_h = mlp_output.reshape(num_peds, num_peds, -1).max(1)[0]
            pool_h.append(frame_pool_h)
            
        pool_h = torch.cat(pool_h, dim = 0)
        return pool_h
    
    
class SocialPooling(nn.Module):
    '''
    Pooling Module meant for capturing adjacent information.
    '''
    def __init__(self, pool_h_dim = 64, batch_norm = True, dropout = 0.0, activation = 'relu',
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
            hidden_states: tensor of shape (num_layers, batch, lstm_h_dim)
            all_pos: tensor of shape (batch, 2)
            seq_start_end: tensor delimiting the start and end of a certain frame
        
        Output:
            pool_output: tensor of shape (batch, self.pool_out_dim)
        '''
        
        hidden_states = hidden_states.squeeze() # Default # of layers = 1, to (batch, lstm_h_dim)
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
    def __init__(self, embedding_dim = 64, encoder_h_dim = 64, decoder_h_dim = 128, 
                 num_layers = 1, batch_norm = True, obs_len = 8, pred_len = 12, dropout = 0.0,
                 activation = 'relu', neighbourhood_size = 2.0, grid_size = 8, 
                 decoder_mlp_dim = 1024, bottleneck_dim = 1024, noise_dim = (0, ), noise_type = 'gaussian',
                 noise_mix_type = 'ped', pool_type = 'pool_net', pool_every_step = True, pool_mlp_dim = 1024):
        
        super(TrajectoryGenerator, self).__init__()
        self.embedding_dim = embedding_dim
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.num_layers = num_layers
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.pool_type = pool_type
        
        self.noise_dim = noise_dim
        self.noise_type = noise_type
        self.noise_mix_type = noise_mix_type
        self.noise_first_dim = 0
        
        self.embedding = nn.Linear(2, embedding_dim)
        self.hidden2pos = nn.Linear(decoder_h_dim, 2)
        
        self.encoder = Encoder(embedding_dim = embedding_dim, hidden_dim = encoder_h_dim, num_layers = num_layers,
                               dropout = dropout)
        self.decoder = Decoder(pred_len = pred_len, embedding_dim = embedding_dim, hidden_dim = decoder_h_dim,
                               bottleneck_dim = bottleneck_dim, mlp_dim = decoder_mlp_dim, num_layers = num_layers,
                               batch_norm = batch_norm, dropout = dropout, pool_type = pool_type,
                               activation = activation, pool_every_step = pool_every_step, 
                               neighbourhood_size = neighbourhood_size, grid_size = grid_size, pool_mlp_dim = pool_mlp_dim)
        
        if pool_type == 'pool_net':
            self.pool_net = PoolNet(embedding_dim = embedding_dim, h_dim = encoder_h_dim, bottleneck_dim = bottleneck_dim,
                                    pool_mlp_dim = pool_mlp_dim, batch_norm = batch_norm, dropout = dropout,
                                    activation = activation)
        else:
            self.pool_net = SocialPooling(pool_h_dim = encoder_h_dim, batch_norm = batch_norm, dropout = dropout,
                                          activation = activation, neighbourhood_size = neighbourhood_size,
                                          grid_size = grid_size, pool_out_dim = None)
        
        
        if noise_dim[0] != 0:
            self.noise_first_dim = noise_dim[0]
        else:
            self.noise_dim = None
            
        if pool_type == 'pool_net':
            input_dim = encoder_h_dim + bottleneck_dim
        else:
            input_dim = encoder_h_dim * 2
        
        if self.extra_mlp_needed():
            extra_mlp_dim_list = [input_dim, decoder_mlp_dim, decoder_h_dim - self.noise_first_dim]
            self.encoder_decoder_mlp = mlp(extra_mlp_dim_list, batch_norm = batch_norm, dropout = dropout,
                                           activation = activation)
        
    
    def extra_mlp_needed(self):
        if (self.encoder_h_dim != self.decoder_h_dim) or (self.noise_dim is not None) or self.pool_type:
            return True
        else:
            return False
        
        
    def add_noise(self, _input, seq_start_end):
        '''
        Input:
            _input: tensor of shape (batch, decoder_h_dim - noise_first_dim)
            seq_start_end: delimiter of different frames
        Output:
            decoder_init_hidden: tensor of shape (batch, decoder_h_dim)
        '''
        if self.noise_dim is None:
            return _input
        
        if self.noise_mix_type == 'ped':
            noise_shape = (_input.size(0), ) + (self.noise_first_dim, )
        else:
            noise_shape = (seq_start_end.size(0), ) + (self.noise_first_dim, )
        
        noise_added = generate_noise(noise_shape, self.noise_type)
        
        if self.noise_mix_type == 'global':
            noise = []
            
            for idx, (start, end) in enumerate(seq_start_end):
                num_peds = end - start
                noise_vec = noise_added[idx].reshape(1, -1)
                noise_vec = noise_vec.repeat(num_peds, 1)
                noise.append(noise_vec)
            
            noise_added = torch.cat(noise, dim = 0)
        
        decoder_init_hidden = torch.cat([_input, noise_added], dim = 1)
        return decoder_init_hidden
        
        
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
        obs_seq_rel_embed = obs_seq_rel.reshape(-1, 2)
        obs_seq_rel_embed = self.embedding(obs_seq_rel_embed)
        obs_seq_rel_embed = obs_seq_rel_embed.reshape(-1, batch, self.embedding_dim)
        
        encoder_final_h = self.encoder(obs_seq_rel_embed)
        
        encoder_last_pos = obs_seq[-1]
        encoder_last_pos_rel = obs_seq_rel[-1]
        
        decoder_init_pool_h = self.pool_net(encoder_final_h, encoder_last_pos, seq_start_end)
        decoder_init_mlp_in = torch.cat([encoder_final_h.squeeze(), decoder_init_pool_h], dim = 1)
        if self.extra_mlp_needed():
            without_noise = self.encoder_decoder_mlp(decoder_init_mlp_in)
        
        with_noise = self.add_noise(without_noise, seq_start_end)
        
        decoder_h = with_noise.unsqueeze(0)
        decoder_c = torch.zeros(self.num_layers, batch, self.decoder_h_dim).cuda()
        state_tuple = (decoder_h, decoder_c)
        
        pred_seq, pred_seq_rel = self.decoder(state_tuple, encoder_last_pos, encoder_last_pos_rel, seq_start_end)
        
        return pred_seq, pred_seq_rel
    
    

class TrajectoryDiscriminator(nn.Module):
    # TODO
    '''
    Discriminator implemented through encoder.
    '''
    def __init__(self, obs_len = 8, pred_len = 12, embedding_dim = 64, h_dim = 64, mlp_dim = 1024,
                 batch_norm = True, dropout = 0.0, activation = 'relu', num_layers = 1, d_type = 'ped'):
        super(TrajectoryDiscriminator, self).__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.embedding_dim = embedding_dim
        self.h_dim = h_dim
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.d_type = d_type
        
        self.encoder = Encoder(embedding_dim = embedding_dim, hidden_dim = h_dim, num_layers = num_layers,
                               dropout = dropout)
        classifier_dim_list = [h_dim, mlp_dim, 1]
        self.classifier = mlp(classifier_dim_list, batch_norm = batch_norm, dropout = dropout, 
                              activation = activation)
        
        if self.d_type == 'global':
            self.pooling = PoolNet(embedding_dim = embedding_dim, 
                                   h_dim = h_dim,
                                   bottleneck_dim = h_dim,
                                   pool_mlp_dim = mlp_dim,
                                   batch_norm = batch_norm,
                                   dropout = dropout,
                                   activation = activation)
        
    def forward(self, traj, traj_rel, seq_start_end = None):
        '''
        Input:
            traj: tensor of shape (obs_len + pred_len, batch, 2)
            traj_rel: tensor of shape (obs_len + pred_len, batch, 2)
            seq_start_end: delimiter
        Output:
            scores: tensor of shape (batch, )
        '''
        final_h = self.encoder(traj_rel)
        
        if self.d_type == 'ped':
            final_h = final_h.squeeze()
        elif self.d_type == 'global':
            final_h = self.pooling(final_h, traj[0], seq_start_end)
        
        scores = self.classifier(final_h)
        
        return scores

        