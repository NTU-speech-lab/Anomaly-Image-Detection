import argparse
import os
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from sklearn.metrics import f1_score, pairwise_distances, roc_auc_score
from sklearn.decomposition import PCA
from scipy.cluster.vq import vq, kmeans
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans
from utils import *
from eva import *

SEED = 878787

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="data/test.npy")
parser.add_argument("--model", default="models/baseline.pth")
parser.add_argument("--output", default="prediction.csv")
parser.add_argument("--problem", type=int, default=2)
sysargs = parser.parse_args()

args = {
    "test_path": sysargs.input,
    "model_path": sysargs.model,
    "pred_path": sysargs.output,
    "problem": sysargs.problem,
    "batch_size": 128,
    "model_type": 'cnn' if os.path.split(sysargs.model)[-1].split('.')[0] == 'best' else 'fcn'
}
args = type('Args', (object, ), args)

evaluator = Evaluator(args)

test = np.load(args.test_path, allow_pickle=True)
test = (test + 1) / 2

if args.problem == 1:
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
        if args.model_type == 'cnn' or args.model_type == 'vae2d':
            img = data[0].transpose(3, 1).cuda()
        else:
            img = data[0].cuda()
        output = model(img)
        if args.model_type == 'cnn' or args.model_type == 'vae2d':
            output = output[0].transpose(3, 1)
        else:
            output = output[0]
        reconstructed.append(output.cpu().detach().numpy())

    reconstructed = np.concatenate(reconstructed, axis=0)
    anomality = np.sqrt(np.sum(np.square(reconstructed - y).reshape(len(y), -1), axis=1))
    print(reconstructed.shape)
    out_img = reconstructed.reshape(reconstructed.shape[0], 32, 32, 3)
    pos = np.argsort(anomality)
    pos = [pos[i] for i in [0, 1, -1, -2]]
    
    fig, ax = plt.subplots(2, 4)
    for i, j in enumerate(pos):
        ax[0, i].imshow(test[j])
        ax[0, i].axis('off')
        ax[1, i].imshow(out_img[j])
        ax[1, i].axis('off')
        ax[1, i].text(0, -1, 'loss: {:.2f}'.format(anomality[j]))

    plt.savefig('img/{}_p1.png'.format(args.model_type))

if args.problem == 2:
    train = np.load("data/train.npy", allow_pickle=True)
    train = (train + 1) / 2

    x = train.reshape(len(train), -1)
    x = torch.tensor(x, dtype=torch.float)

    y = test.reshape(len(test), -1)
    y = torch.tensor(y, dtype=torch.float)

    model = torch.load("models/baseline.pth", map_location='cuda')
    model.eval()

    ex = model(x.cuda())[-1].cpu().detach().numpy()
    ey = model(y.cuda())[-1].cpu().detach().numpy()
    
    kmeans_x = MiniBatchKMeans(n_clusters=3, batch_size=64).fit(ex)
    y_cluster = kmeans_x.predict(ey)
    y_dist = np.sum(np.square(kmeans_x.cluster_centers_[y_cluster] - ey), axis=1)        
    y_pred = y_dist
    print("{}: {}".format(3, evaluator.score(y_pred)))

    with open('knn_base.csv', 'w') as f:
        f.write('id,anomaly\n')
        for i in range(len(y_pred)):
            f.write('{},{}\n'.format(i+1, y_pred[i]))

    pca = PCA(n_components=3).fit(ex)

    y_projected = pca.transform(ey)
    y_reconstructed = pca.inverse_transform(y_projected)
    dist = np.sqrt(np.sum(np.square(y_reconstructed - ey).reshape(len(y), -1), axis=1))

    y_pred = dist
    print("PCA: {}".format(evaluator.score(y_pred)))

    with open('pca_base.csv', 'w') as f:
        f.write('id,anomaly\n')
        for i in range(len(y_pred)):
            f.write('{},{}\n'.format(i+1, y_pred[i]))

    x = train
    x = torch.tensor(x, dtype=torch.float).transpose(3, 1)
    
    y = test
    y = torch.tensor(y, dtype=torch.float).transpose(3, 1)

    model = torch.load("models/best.pth", map_location='cuda')
    model.eval()

    ex = model(x.cuda())[-1].view(len(train), -1).cpu().detach().numpy()
    ey = model(y.cuda())[-1].view(len(test), -1).cpu().detach().numpy()    
    
    kmeans_x = MiniBatchKMeans(n_clusters=3, batch_size=64).fit(ex)
    y_cluster = kmeans_x.predict(ey)
    y_dist = np.sum(np.square(kmeans_x.cluster_centers_[y_cluster] - ey), axis=1)        
    y_pred = y_dist

    with open('knn_best.csv', 'w') as f:
        f.write('id,anomaly\n')
        for i in range(len(y_pred)):
            f.write('{},{}\n'.format(i+1, y_pred[i]))

    print("{}: {}".format(3, evaluator.score(y_pred)))

    pca = PCA(n_components=3).fit(ex)

    y_projected = pca.transform(ey)
    y_reconstructed = pca.inverse_transform(y_projected)  
    dist = np.sqrt(np.sum(np.square(y_reconstructed - ey).reshape(len(y), -1), axis=1))
    
    y_pred = dist

    with open('pca_best.csv', 'w') as f:
        f.write('id,anomaly\n')
        for i in range(len(y_pred)):
            f.write('{},{}\n'.format(i+1, y_pred[i]))
    print("PCA: {}".format(evaluator.score(y_pred)))

if args.problem == 3:
    y = test.reshape(len(test), -1)
    data = torch.tensor(y, dtype=torch.float)
    
    emb = TSNE(n_components=2, random_state=0).fit_transform(y)
    plt.scatter(emb[:, 0], emb[:, 1], s=1)
    plt.savefig("img/origin_p3.png")
    plt.close()
    print('origin finish')

    model = torch.load("models/baseline.pth", map_location='cuda')
    model.eval()

    emb = model(data.cuda())[-1].cpu().detach().numpy()
    emb = TSNE(n_components=2, random_state=0).fit_transform(emb)
    plt.scatter(emb[:, 0], emb[:, 1], s=1)
    plt.savefig("img/baseline_p3.png")
    plt.close()

    y = test
    data = torch.tensor(y, dtype=torch.float).transpose(3, 1)

    model = torch.load("models/best.pth", map_location='cuda')
    model.eval()

    emb = model(data.cuda())[-1].view(len(data), -1).cpu().detach().numpy()
    emb = TSNE(n_components=2, random_state=0).fit_transform(emb)
    plt.scatter(emb[:, 0], emb[:, 1], s=1)
    plt.savefig("img/best_p3.png")
    plt.close()

print("--- %s seconds ---" % (time.time() - start_time))