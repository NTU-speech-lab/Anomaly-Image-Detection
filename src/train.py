import argparse
import os
import numpy as np
import torch
import time
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from model import *
from utils import *

SEED = 878787

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="data/train.npy")
parser.add_argument("--output", default="models/baseline.pth")
sysargs = parser.parse_args()

args = {
    "train_path": sysargs.input,
    "model_path": sysargs.output,
    "mode": os.path.split(sysargs.output)[-1].split('.')[0],
    "num_epochs": 1000,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "model_type": 'fcn',
    "local_mode": False,
    "criterion": nn.MSELoss()
}
args = type('Args', (object, ), args)

train = np.load(args.train_path, allow_pickle=True)
train = (train + 1) / 2
print(np.max(train))

evaluator = None
if args.local_mode:
    from eva import *
    evaluator = Evaluator(args)

print(train.shape)

if args.mode == 'baseline':
    args.model_type = 'fcn'
if args.mode == 'best':
    args.model_type = 'cnn'

x = train
if args.model_type == 'fcn' or args.model_type == 'vae':
    x = x.reshape(len(x), -1)

data = torch.tensor(x, dtype=torch.float)
train_dataset = TensorDataset(data)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

model_classes = {'fcn':fcn_autoencoder(), 'cnn':conv_autoencoder(), 'vae':VAE(), 'vae2d':VAE2d()}

model = model_classes[args.model_type].cuda()
criterion = args.criterion
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

best_loss = np.inf
hs = 0
h_epoch = 0
model.train()
for epoch in range(args.num_epochs):
    model.train()
    avg_loss = 0
    for data in train_dataloader:
        if args.model_type == 'cnn' or args.model_type == 'vae2d':
            img = data[0].transpose(3, 1).cuda()
        else:
            img = data[0].cuda()
        # ===================forward=====================
        output = model(img)
        if args.model_type == 'vae':
            loss = loss_vae(output[0], img, output[1], output[2], criterion)
        elif args.model_type == 'vae2d':
            loss = loss_vae(output[0], img, output[1], output[2], criterion, is2d=True)
        else:
            loss = criterion(output[0], img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() * len(data[0])

    avg_loss /= len(train_dataloader)
    # ===================save====================
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model, args.model_path)
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, args.num_epochs, avg_loss))

print("--- %s seconds ---" % (time.time() - start_time))
