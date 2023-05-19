import jittor as jt
from jittor import nn
from jittor import models
from jittor import Module
from jittor import optim
from jittor import init
from jittor import transform
from jittor.dataset import Dataset, DataLoader
from jittor.dataset.mnist import MNIST
import os
import numpy as np
import sys
import random
import math
import gzip
from PIL import Image
from jittor_utils.misc import download_url_to_local
import matplotlib
import matplotlib.pyplot as plt


jt.flags.use_cuda = 1
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
CUDA_VISIBLE_DEVICES = '1'


'''
Transformation of the deprecated dataset
'''
trans = transform.Compose([
    transform.RandomResizedCrop(size = (28, 28), scale = (0.5, 1.0), ratio = (0.75, 1.33333)),
    transform.ToTensor()
])


def MLP(mlp_list, batch_norm = True, dropout = 0.0, activation = 'relu'):
    net = []
    for i, (in_dim, out_dim) in enumerate(zip(mlp_list[:-1], mlp_list[1:])):
        net.append(nn.Linear(in_dim, out_dim))
        if batch_norm and i != len(mlp_list) - 2:
            net.append(nn.BatchNorm(out_dim))
        if activation == 'relu' and i != len(mlp_list) - 2:
            net.append(nn.ReLU())
        net.append(nn.Dropout(p = dropout))
    
    net = nn.Sequential(*net)
    return net


def weights_init(m): 
    if isinstance(m, nn.Conv2d): 
        init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
        
        
        
class MNIST(Dataset):
    def __init__(self, data_root="./mnist_data/", train=True ,download=True, batch_size=1, shuffle=False):
        # if you want to test resnet etc you should set input_channel = 3, because the net set 3 as the input dimensions
        super(MNIST, self).__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.is_train = train
        if download == True:
            self.download_url()

        filesname = [
                "train-images-idx3-ubyte.gz",
                "t10k-images-idx3-ubyte.gz",
                "train-labels-idx1-ubyte.gz",
                "t10k-labels-idx1-ubyte.gz"
        ]
        self.mnist = {}
        if self.is_train:
            with gzip.open(data_root + filesname[0], 'rb') as f:
                self.mnist["images"] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28, 28)
                # transformed = trans(self.mnist["images"].reshape(-1, 1, 28, 28))
                # print(self.mnist["images"].shape)
                # print(transformed.shape)
                # self.mnist["images"] = np.concatenate([self.mnist["images"], transformed], axis = 0)
            with gzip.open(data_root + filesname[2], 'rb') as f:
                self.mnist["labels"] = np.frombuffer(f.read(), np.uint8, offset=8)
                # self.mnist["labels"] = np.concatenate([self.mnist["labels"], self.mnist["labels"]], axis = 0)
        else:
            with gzip.open(data_root + filesname[1], 'rb') as f:
                self.mnist["images"] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28, 28)
                # transformed = trans(self.mnist["images"].reshape(-1, 1, 28, 28))
                # self.mnist["images"] = np.concatenate([self.mnist["images"], transformed], axis = 0)
            with gzip.open(data_root + filesname[3], 'rb') as f:
                self.mnist["labels"] = np.frombuffer(f.read(), np.uint8, offset=8)
                # self.mnist["labels"] = np.concatenate([self.mnist["labels"], self.mnist["labels"]], axis = 0)
        assert(self.mnist["images"].shape[0] == self.mnist["labels"].shape[0])
        self.total_len = self.mnist["images"].shape[0]
        # this function must be called
        self.set_attrs(batch_size = self.batch_size, total_len=self.total_len, shuffle= self.shuffle)
    def __getitem__(self, index):
        img = Image.fromarray (self.mnist['images'][index]) 
        img = np.array (img)
        img = img[np.newaxis, :]
        return np.array((img / 255.0), dtype = np.float32), self.mnist['labels'][index]

    def download_url(self):
        resources = [
            ("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
            ("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
            ("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
            ("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
        ]

        for url, md5 in resources:
            filename = url.rpartition('/')[2]
            download_url_to_local(url, filename, self.data_root, md5)



class RNN(Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.embedding = nn.Linear(28, 32)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(32, 32)
        self.fc = nn.Linear(32, 10)
        
    def execute(self, x):
        '''
        x: jittor array of shape (batch, 28, 28)
        '''
        batch = x.shape[0]
        x = x.reshape(-1, 28)
        x = self.relu(self.embedding(x))
        x = x.reshape(-1, 28, 32)
        x = x.transpose(1, 0, 2)
        x, (h, c) = self.lstm(x)
        x = x[-1, :, :]
        x = self.fc(x)
        return x
    
    
model = RNN()
model.apply(weights_init)
# init.kaiming_normal_(model.conv1.weight)
# init.kaiming_normal_(model.conv2.weight)
# init.kaiming_normal_(model.fc[0].weight)

model.cuda()
model.train()
lr = 0.03
momentum = 0.9
weight_decay = 1e-4
epoch = 400
optimizer = optim.SGD(model.parameters(), lr = lr, momentum = momentum, weight_decay = weight_decay)

train_loader = MNIST(train = True, batch_size = 64, shuffle = True)
test_loader = MNIST(train = False, batch_size = 1, shuffle = False)

'''
Modification and of the training set
'''
# print(train_loader.mnist['images'].shape)
images_masked = train_loader.mnist['images'][train_loader.mnist['labels'] <= 4, :, :] 
images_rest = train_loader.mnist['images'][train_loader.mnist['labels'] >= 5, :, :]
labels_masked = train_loader.mnist['labels'][train_loader.mnist['labels'] <= 4] 
labels_rest = train_loader.mnist['labels'][train_loader.mnist['labels'] >= 5]
images_masked = images_masked[: int(0.1 * len(images_masked)), :, :]
labels_masked = labels_masked[: int(0.1 * len(labels_masked))]
# train_loader.mnist['images'] = np.concatenate([images_masked, images_rest], axis = 0)
# train_loader.mnist['labels'] = np.concatenate([labels_masked, labels_rest], axis = 0)
# print(train_loader.mnist['labels'].shape)


'''
Compensation for the modification through data augmentation
'''
to_append = 9
transformed_seq = []
new_labels_seq = []
for idx in range(len(images_masked)):
    original_img = images_masked[idx]
    for i in range(to_append):
        transformed = trans(original_img)
        # print(transformed.shape)
        transformed_seq.append(transformed)
        new_labels_seq.append(labels_masked[idx])
        
transformed_seq = np.array(transformed_seq)
transformed_seq = transformed_seq.reshape(-1, 28, 28)
new_labels_seq = np.array(new_labels_seq)
train_loader.mnist['images'] = np.concatenate([images_masked, transformed_seq, images_rest], axis = 0)
train_loader.mnist['labels'] = np.concatenate([labels_masked, new_labels_seq, labels_rest], axis = 0)
train_loader.total_len = len(train_loader.mnist['labels'])
train_loader.set_attrs()
print(train_loader.mnist['labels'].shape)


'''
----------
Training
----------
'''
losses = []

for t in range(epoch):
    loss_sum = 0
    for i, (images, labels) in enumerate(train_loader):
        # print(labels)
        optimizer.zero_grad()
        score = model(images)
        loss = nn.cross_entropy_loss(score, labels)
        # if (i == 0 or i == 1) and t == 0:
        #     print(images.shape, labels.shape)
        #     print(score.shape)
        loss_sum += loss
        optimizer.backward(loss)
        optimizer.step()
    
    loss_sum = loss_sum / len(train_loader)
    if t % 10 == 0:
        print('Epoch: %d Loss: %.4f' % (t, loss_sum))
    losses.append(loss_sum)
    
losses = np.array(losses)

plt.figure()
plt.plot(np.arange(epoch), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss(CEL)')
plt.title('RNN result (compensated) Learning rate: %.2f' % lr)

'''
----------
Evaluation
----------
'''
model.eval()
total_correct = 0
total_images = 0

for i, (images, labels) in enumerate(test_loader):
    # if i == 0 or i == 1:
    #     print(images.shape, labels.shape)
    bs = images.shape[0]
    pred = model(images)
    pred_label = np.argmax(pred.numpy(), axis = 1)
    total_correct += np.sum(pred_label == labels.numpy())
    total_images += bs
    # loss = nn.cross_entropy_loss(pred, labels)
    # loss_sum += loss
    
acc = 100.0 * total_correct / total_images 
print('Avg Test Accuracy: %.4f %% Learining Rate: %.3f Epoch: %d' % (acc, lr, epoch))

plt.show()