import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import PIL.Image as Image
import cPickle as pickle
    

def plot_loss(losses, mini=None, maxi=None):
    loss_np = np.array(losses)
    plt.figure(figsize=(10,6))
    plt.plot(loss_np[:,0])
    plt.plot(loss_np[:,1])
    plt.legend(["Training loss", "Validation loss"])
    if (mini is not None):
        plt.ylim(mini, maxi)


def plot_features(W, cmap=plt.cm.seismic, shared_colorbar=True, name='decoder.png'):
    print "==> visualizing features"

    cnt = int(np.sqrt(W.shape[1]) + 1 - 1e-6)
    if W.shape[0] == 784:
        fig, ax = plt.subplots(cnt, cnt, figsize=(10,10))
    else:
        fig, ax = plt.subplots(cnt, cnt, figsize=(18,10))
    
    if cmap==plt.cm.seismic:
        mini = -np.max(np.abs(W))
        maxi = -mini
    else:
        mini = np.min(W)
        maxi = np.max(W)
    
    for i in range(cnt):
        for j in range(cnt):
            if i*cnt+j >= W.shape[1]:
                break
            if shared_colorbar:
                im = ax[i][j].imshow(W[:,i*cnt+j].reshape((28,-1)), cmap=cmap, vmin=mini, vmax=maxi)
            else:
                im = ax[i][j].imshow(W[:,i*cnt+j].reshape((28,-1)), cmap=cmap)
            ax[i][j].get_xaxis().set_visible(False)
            ax[i][j].get_yaxis().set_visible(False)
    
    if shared_colorbar:
        fig.colorbar(im, ax=ax.ravel().tolist())

    fig.savefig(name)


def plot_features_clip(W):
    """
    Plot features by cliping all big values (which are on the borders) usually
    """
    print "==> visualizing features (clipped)"

    cnt = int(np.sqrt(W.shape[1]) + 1 - 1e-6)
    fig, ax = plt.subplots(cnt, cnt, figsize=(10,10))
    
    mean = np.mean(np.abs(W))
    
    for i in range(cnt):
        for j in range(cnt):
            if i*cnt+j >= W.shape[1]:
                break
            mas = np.clip(W[:,i*cnt+j].reshape((28,28)), -3*mean, +3*mean)
            im = ax[i][j].imshow(mas, cmap=plt.cm.seismic, vmin=-3*mean, vmax=+3*mean)
            ax[i][j].get_xaxis().set_visible(False)
            ax[i][j].get_yaxis().set_visible(False)
    
    fig.colorbar(im, ax=ax.ravel().tolist())


def plot_features_remove(W):
    """
    Plot features of by removing all big values (which are on the borders) usually
    """
    print "==> visualizing features (removed)"

    cnt = int(np.sqrt(W.shape[1]) + 1 - 1e-6)
    fig, ax = plt.subplots(cnt, cnt, figsize=(10,10))
    
    mean = np.mean(np.abs(W))
    
    for i in range(cnt):
        for j in range(cnt):
            if i*cnt+j >= W.shape[1]:
                break
            
            mas = 1.0 * W[:,i*cnt+j].reshape((28,28))
            mas[np.abs(mas) > 3*mean] = 0
            im = ax[i][j].imshow(mas, cmap=plt.cm.seismic, vmin=-3*mean, vmax=+3*mean)
            ax[i][j].get_xaxis().set_visible(False)
            ax[i][j].get_yaxis().set_visible(False)
    
    fig.colorbar(im, ax=ax.ravel().tolist())
    
    
def plot_features_reverse(W):
    print "==> visualizing features in reverse direction"

    A = 1
    B = W.shape[1]
    
    for a in range(1, W.shape[1] + 1):
        if W.shape[1] % a == 0:
            b = W.shape[1] / a
            if abs(a-b) < abs(A-B):
                A,B = a,b
                
    fig, ax = plt.subplots(28, 28, figsize=(16,16))
    mini = -np.max(np.abs(W))
    maxi = -mini
    
    for i in range(28):
        for j in range(28):
            k = 28*i + j
            im = ax[i][j].imshow(W[k, :].reshape((A,B)), cmap=plt.cm.seismic, vmin=mini, vmax=maxi)
            ax[i][j].get_xaxis().set_visible(False)
            ax[i][j].get_yaxis().set_visible(False)
    
    fig.colorbar(im, ax=ax.ravel().tolist())


def plot_z_values(Z):
    # (n_examples, hidden_values)
    fig, ax = plt.subplots(5,5, figsize=(16,16))
    for i in range(5):
        for j in range(5):
            ax[i][j].hist(Z[:, i*20+j], bins=50);
            ax[i][j].title.set_text("std: %.3f" % np.std(Z[:, i*20+j]))
            #ax[i][j].get_xaxis().set_visible(False)
            #ax[i][j].get_yaxis().set_visible(False)
