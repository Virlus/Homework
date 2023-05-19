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
    

class TrajectoryGenerator(nn.Module):
    '''
    The combination of encoder and decoder, exploited to predict the future trajectories.
    '''
    def __init__(self, input_dim = 2, embedding_dim = 64, encoder_h_dim = 64, decoder_h_dim = 64, 
                 num_layers = 1, batch_norm = 1, obs_len = 8, pred_len = 12, dropout = 0.0):
        
        super(TrajectoryGenerator, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.num_layers = num_layers
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.dropout = dropout
        

        self.embedding = nn.Linear(input_dim, embedding_dim)
        self.hidden2pos = nn.Linear(decoder_h_dim, 2)
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = encoder_h_dim, num_layers = num_layers,
                            dropout = dropout)
        
    
    
    def get_hidden(self, batch):
        return (torch.zeros(self.num_layers, batch, self.encoder_h_dim).cuda(), 
                torch.zeros(self.num_layers, batch, self.encoder_h_dim).cuda())
        
        
    def forward(self, obs_seq, obs_seq_rel):
        '''
        Input:
            obs_seq: tensor of shape (obs_len, batch, 2)
            obs_seq_rel: tensor of shape (obs_len, batch, 2)
        
        Output:
            pred_seq: tensor of shape (pred_len, batch, 2)
            pred_seq_rel: tensor of shape (pred_len, batch, 2)
        '''
        batch = obs_seq_rel.shape[1]
        obs_seq_rel_embed = obs_seq_rel.reshape(-1, self.input_dim)
        obs_seq_rel_embed = self.embedding(obs_seq_rel_embed)
        obs_seq_rel_embed = obs_seq_rel_embed.reshape(-1, batch, self.embedding_dim)
        
        _, state = self.lstm(obs_seq_rel_embed, self.get_hidden(batch))
        
        encoder_last_pos = obs_seq[-1]
        encoder_last_pos_rel = obs_seq_rel[-1]
        #print(encoder_last_pos.shape, encoder_last_pos_rel.shape, encoder_state[0].shape,
              #encoder_state[1].shape)
        
        state_tuple = state
        
        decoder_input = self.embedding(encoder_last_pos_rel)
        decoder_input = decoder_input.unsqueeze(0)
        last_pos = encoder_last_pos
        
        pred_seq = []
        pred_seq_rel = []
        
        for _ in range(self.pred_len):
            output, state_tuple = self.lstm(decoder_input, state_tuple)
            output = output.reshape(-1, self.decoder_h_dim)
            rel_pos = self.hidden2pos(output)
            curr_pos = rel_pos + last_pos
            pred_seq.append(curr_pos)
            pred_seq_rel.append(rel_pos)
            last_pos = curr_pos
            
            decoder_input = self.embedding(rel_pos)
            decoder_input = decoder_input.unsqueeze(0)
        
        pred_seq = torch.stack(pred_seq, dim = 0)
        pred_seq_rel = torch.stack(pred_seq_rel, dim = 0)
        #print(pred_seq.shape, pred_seq_rel.shape)
        
        return pred_seq, pred_seq_rel
    

        