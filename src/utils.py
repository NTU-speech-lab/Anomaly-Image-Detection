from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def convert(x):
    x = (x + 1) / 2
    plt.imshow(x)
    plt.show()

def show(x):
    plt.imshow(x)
    plt.show()

def plot_all(x):
    fig, ax = plt.subplots(10, 10)
    for i in range(x.shape[0]):
        pos = i % 100
        r = pos // 10
        c = pos % 10
        ax[r, c].imshow((x[i] + 1) / 2)
        ax[r, c].axis('off')
        if pos == 99:
            plt.savefig("./picture/test/{}.png".format(i // 100))
            plt.close()
            fig, ax = plt.subplots(10, 10)

def run():
    train = np.load("data/test.npy", allow_pickle=True)
    plot_all(train)