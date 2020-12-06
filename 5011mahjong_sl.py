import warnings
warnings.filterwarnings('ignore')
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import random
from tqdm import tqdm
import numpy as np
import bloscpack as bp
from sklearn import metrics
import time
from datetime import timedelta
from torch.utils.data import Dataset, DataLoader
import numpy as np
from functools import partial
from collections import Counter

master_X = bp.unpack_ndarray_from_file('input_X_0_100136.nosync.blp')
master_Y = bp.unpack_ndarray_from_file('input_Y_0_100136.nosync.blp')
master_Y = np.where(master_Y==1)[1]

# split training and validation set
n = len(master_X)
train_size = int(0.8*n)
train_idx = random.sample(range(n),train_size)
val_idx = [i for i in range(n) if i not in train_idx]

train_X = torch.from_numpy(master_X[train_idx,:,:,:]).type(torch.FloatTensor)
train_Y = torch.from_numpy(master_Y[train_idx]).type(torch.LongTensor)
val_X = torch.from_numpy(master_X[val_idx,:,:,:]).type(torch.FloatTensor)
val_Y = torch.from_numpy(master_Y[val_idx]).type(torch.LongTensor)

# Transform into dataloader
train_dataset = TensorDataset(train_X, train_Y)
val_dataset = TensorDataset(val_X, val_Y)

train_dataloader = DataLoader(train_dataset,batch_size=128,shuffle=True,num_workers=2)
val_dataloader = DataLoader(val_dataset,batch_size=128,shuffle=False,num_workers=2)


########################################
class SamePadConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)


conv3x1 = partial(SamePadConv2d, kernel_size=(3, 1))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.layer1 = self.make_layer(in_channels)
        self.layer2 = self.make_layer(in_channels)

    def make_layer(self, in_channels, dropout_prob=0.5):
        layer = nn.Sequential(
            conv3x1(in_channels, in_channels),
            nn.BatchNorm2d(256),
            nn.Dropout2d(dropout_prob),
            nn.LeakyReLU()
        )
        return layer

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out += x
        return out


class MJResNet50(nn.Module):
    def __init__(self, history_len=4, n_cls=34, n_residuals=50):
        super().__init__()
        self.net = self.create_model(50, n_residuals, n_cls)  # 改了第一个参数

    def forward(self, x):
        return self.net(x)

    def create_model(self, in_channels, n_residuals, n_cls):
        # First layer
        module_list = nn.ModuleList([
            conv3x1(in_channels, 256),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.5),
            nn.LeakyReLU()
        ])
        # Adding residual blocks
        for layer_i in range(n_residuals):
            module_list.append(ResidualBlock(256))

        # Flatten & then fc layers
        module_list.append(nn.Flatten())
        out_feat = 1024
        module_list += nn.ModuleList([
            *self.linear_block(256 * 34 * 4, 1024, dropout_prob=0.2),  # 改了全连接层
            *self.linear_block(1024, 256, dropout_prob=0.2),
            nn.Linear(256, n_cls)
        ])

        return nn.Sequential(*module_list)

    def linear_block(self, n_feat, out_feat, dropout_prob=0.5):
        block = nn.ModuleList([
            nn.Linear(n_feat, out_feat),
            nn.BatchNorm1d(out_feat),
            nn.Dropout(dropout_prob),
            nn.LeakyReLU()
        ])
        return block
########################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = MJResNet50()
net.to(device)

import torch.optim as optim
def get_time_dif(start_time):
  end_time = time.time()
  time_dif = end_time - start_time
  return timedelta(seconds=int(round(time_dif)))

def evaluate(model, data_loader):
  model.eval()
  loss_total = 0
  predict_all = np.array([], dtype=int)
  labels_all = np.array([], dtype=int)
  with torch.no_grad():
    for inputs, labels in data_loader:
      inputs, labels = inputs.to(device), labels.to(device)
      outputs = model(inputs)
      loss = F.cross_entropy(outputs, labels)
      loss_total += loss
      labels = labels.data.cpu().numpy()
      predic = torch.max(outputs.data,1)[1].cpu().numpy()
      labels_all = np.append(labels_all, labels)
      predict_all = np.append(predict_all, predic)
  acc = metrics.accuracy_score(labels_all, predict_all)
  recall = metrics.recall_score(labels_all, predict_all,average='macro')
  f1 = metrics.f1_score(labels_all, predict_all,average='macro')
  return acc, recall, f1, loss_total/len(data_loader)

def train(model,train_loader,val_loader,learning_rate=0.0025,
    num_epochs=200,save_path='best_sl.ckpt'):
  start_time = time.time()
  model.train()
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  total_batch = 0
  val_best_loss = float('inf')
  last_improve = 0
  flag = False
  model.train()

  for epoch in range(num_epochs):
    running_loss = 0.0
    print('Epoch[{}/{}]'.format(epoch+1, num_epochs))
    for i,data in enumerate(train_loader,0):
      inputs, labels = data
      inputs, labels = inputs.to(device), labels.to(device)
      optimizer.zero_grad()

      outputs = model(inputs)
      loss = criterion(outputs,labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      if i % 50 == 49:
        true = labels.data.cpu()
        predic = torch.max(outputs.data,1)[1].cpu()
        train_acc = metrics.accuracy_score(true,predic)
        val_acc, val_recall, val_f1, val_loss = evaluate(model,val_loader)
        if val_loss < val_best_loss:
          val_best_loss = val_loss
          torch.save(model.state_dict(),save_path)
          improve = '*'
          last_improve = total_batch
        else:
          improve = ''
        time_dif = get_time_dif(start_time)
        msg = 'Iter: {0:>6}, Train Loss:{1:>5.4}, Train Acc:{2:6.2%}, Val Loss:{3:5.4}, Val Acc:{4:6.2%}, Val Recall:{5:6.2%} Val F1:{6:6.2%}, Time:{7} {8}'
        print(msg.format(total_batch,running_loss / 20,train_acc,val_loss,val_acc,val_recall,val_f1,time_dif,improve))
        model.train()
        running_loss = 0.0
      total_batch += 1
      # if total_batch - last_improve > 10000:
      #   print("No optimization for a long time, auto-stopping...")
      #   flag = True
      #   break
    # if flag:
    #   break

model = MJResNet50().to(device)
train(model, train_dataloader, val_dataloader,learning_rate=0.001)

