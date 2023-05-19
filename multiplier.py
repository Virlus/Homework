import jittor as jt
from jittor import nn
from jittor import models
from jittor import Module
from jittor import optim
from jittor import init
from jittor.dataset import Dataset, DataLoader
import os
    
    
def MLP(mlp_list, batch_norm = True, dropout = 0.0, activation = 'relu'):
    net = []
    for in_dim, out_dim in zip(mlp_list[:-1], mlp_list[1:]):
        net.append(nn.Linear(in_dim, out_dim))
        if batch_norm:
            net.append(nn.BatchNorm(out_dim))
        if activation == 'relu':
            net.append(nn.ReLU())
        net.append(nn.Dropout(p = dropout))
    
    net = nn.Sequential(*net)
    return net


class Multiplier_Dataset(Dataset):
    def __init__(self, num_pairs):
        super(Multiplier_Dataset, self).__init__()
        
        self.raw = jt.randint(0, 10, shape = (num_pairs, 2))
        self.result = self.raw[:, 0] * self.raw[:, 1]
        
    def __len__(self):
        return len(self.raw)
        
    def __getitem__(self, k):
        return self.raw[k], self.result[k]


class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.mlp = MLP([2, 10, 10, 10, 1])
        
    def execute(self, x):
        x = self.mlp(x)
        return x
    
    

jt.flags.use_cuda = 1
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


train_dataset = Multiplier_Dataset(4000)
train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
test_dataset = Multiplier_Dataset(1000)


model = Model()
model.cuda()
lr = 0.03
epoch = 200
optimizer = optim.Adam(model.parameters(), lr = lr)
loss_fn = nn.MSELoss()


model.train()
for i in range(epoch):
    loss_sum = 0
    for raw, result in train_loader:
        pred = model(raw)
        loss = loss_fn(pred, result)
        loss_sum += loss
        optimizer.step(loss)
    
    loss_sum = loss_sum / len(train_loader)
    if i % 10 == 0:
        print('Epoch: %d Loss: %.3f' % (i + 1, loss_sum))
        
        
model.eval()
loss_sum = 0
for data, label in test_dataset:
    pred = model(data)
    loss = loss_fn(pred, label)
    loss_sum += loss
loss_sum = loss_sum / len(test_dataset)

print('Avg test MSE loss: %.3f' % loss_sum)

    
    


        




