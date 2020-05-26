import argparse
import os
import numpy as np
import torch
import time
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from sklearn.metrics import f1_score, pairwise_distances, roc_auc_score
from PIL import Image
from utils import *

np.random.seed(0x5EED)
torch.manual_seed(0x5EED)
torch.cuda.manual_seed_all(0x5EED)
torch.backends.cudnn.deterministic = True

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="data/test.npy")
parser.add_argument("--model", default="models/baseline.pth")
parser.add_argument("--output", default="prediction.csv")
sysargs = parser.parse_args()

args = {
    "test_path": sysargs.input,
    "model_path": sysargs.model,
    "pred_path": sysargs.output,
    "mode": os.path.split(sysargs.model)[-1].split('.')[0],
    "num_epochs": 1000,
    "batch_size": 128,
    "learning_rate": 1e-5,
    "model_type": 'vae'
}
args = type('Args', (object, ), args)

test = np.load(args.test_path, allow_pickle=True)

if args.mode == 'baseline':
    if args.model_type == 'fcn' or args.model_type == 'vae':
        y = test.reshape(len(test), -1)
    else:
        y = test
        
    data = torch.tensor(y, dtype=torch.float)
    test_dataset = TensorDataset(data)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    model = torch.load(args.model_path, map_location='cuda')

    model.eval()
    reconstructed = list()
    for i, data in enumerate(test_dataloader): 
        if args.model_type == 'cnn':
            img = data[0].transpose(3, 1).cuda()
        else:
            img = data[0].cuda()
        output = model(img)
        if args.model_type == 'cnn':
            output = output.transpose(3, 1)
        elif args.model_type == 'vae':
            output = output[0]
        reconstructed.append(output.cpu().detach().numpy())

    reconstructed = np.concatenate(reconstructed, axis=0)
    anomality = np.sqrt(np.sum(np.square(reconstructed - y).reshape(len(y), -1), axis=1))
    y_pred = anomality
    with open(args.pred_path, 'w') as f:
        f.write('id,anomaly\n')
        for i in range(len(y_pred)):
            f.write('{},{}\n'.format(i+1, y_pred[i]))
    # score = roc_auc_score(y_label, y_pred, average='micro')
    # score = f1_score(y_label, y_pred, average='micro')
    # print('auc score: {}'.format(score))

print("--- %s seconds ---" % (time.time() - start_time))