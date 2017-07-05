"""Utilities for visualizing Keras autoencoders."""
import matplotlib
# Force matplotlib to not use any Xwindows backend.
import sys
#sys.path.insert(0, '../three-letter-mnist/')
import emnist_words
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras import backend as K
from keras.datasets import mnist, cifar10
from keras.layers import merge, Input, Dense
from keras.models import Model
import os
import errno
import cPickle
from sklearn.svm import LinearSVC
from sklearn.cross_validation import cross_val_score
import enchant



def get_emnist_lettercomb(letters, sample='uniform', seed=0):
    #Returns EMNIST 3-letter-word images with given letters dictionary {}:  letters[0] = 1st letter, etc. 
    d = enchant.Dict("en_US")
    words = []
    test_words = []
    for i in letters[0]:
        for j in letters[1]:
            for k in letters[2]:
                if d.check(str(i+j+k)):
                    words.append(str(i+j+k))
                else:
                    test_words.append(str(i+j+k))
    words = np.array(words)
    test_words = np.array(test_words)
    n_words = words.shape[0]
    n_words_test = test_words.shape[0]
    prob = np.array([1./n_words]*n_words)
    probt = np.array([1./n_words_test]*n_words_test)
    per_word = np.array(np.multiply(n_words,prob)).astype(int)
    per_word_test = np.array(np.multiply(n_words_test,probt)).astype(int)
    x_train, x_test, y_train, y_test = emnist_data(per_word = per_word, per_word_test = per_word_test, sample = sample, words = words, test_words = test_words, seed = 0)
    return x_train, x_test, y_train, y_test

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def emnist_seq_data():
    original_dim = 28*84
    x_train, y_train = emnist_words.get_data(data = 'train', addl_path = '../three-letter-mnist')
    x_test, y_test = emnist_words.get_data(data = 'test', addl_path = '../three-letter-mnist')
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    if x_train.shape[1] != original_dim:
        x_train = x_train.reshape((x_train.shape[0], original_dim))
    if x_test.shape[1] != original_dim:
        x_test = x_test.reshape((x_test.shape[0], original_dim))
    return x_train, x_test, y_train, y_test


def emnist_data(per_word = 500, per_word_test = 500, sample = 'normal', words = True,  test_words = True, original_dim = 28*84, seed = 0, set_letters = True):
    x_train, y_train, letters_ind = emnist_words.get_data(per_word = per_word, data = 'train', addl_path = '../three-letter-mnist', get_tw = words, sample = sample, seed = seed, set_letters = set_letters)
    x_test, y_test, a = emnist_words.get_data(per_word = per_word_test, data = 'train', addl_path = '../three-letter-mnist', get_tw = test_words, sample = sample, seed = seed, set_letters = letters_ind)
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    if x_train.shape[1] != original_dim:
        x_train = x_train.reshape((x_train.shape[0], original_dim))
    if x_test.shape[1] != original_dim:
        x_test = x_test.reshape((x_test.shape[0], original_dim))
    return x_train, x_test, y_train, y_test

def mnist_data(onehot=False):
    original_dim = 784
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), original_dim))
    x_test = x_test.reshape((len(x_test), original_dim))
    # Clever one hot encoding trick
    if onehot:
        y_train = np.eye(10)[y_train]
        y_test = np.eye(10)[y_test]
    return x_train, x_test, y_train, y_test


def cifar_data():
    dims = (3, 32, 32)
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    return X_train, X_test, y_train, y_test


def mnist_var(dataset='mnist_background_random'):
    """['mnist', 'rectangles', 'mnist_background_images', 'rectangles_im',
            'mnist_all_background_images_rotation_normalized', 'mnist_background_random',
            'convex', 'mnist_rotation_new']"""
    dir = '/Users/gregv/deepcloud/tf_corex/data/{}/{}_{}.amat'
    self.raw_train = np.loadtxt(dir.format(dataset, dataset, 'test'))
    self.raw_test = np.loadtxt(dir.format(dataset, dataset, 'train'))
    data_train = self.raw_train[:, :-1].reshape((-1, 28, 28, 1))
    data_test = self.raw_test[:, :-1].reshape((-1, 28, 28, 1))


def patches_data(size=16, test_size=10000):
    # Data from the BSDS 500 database. Grayscale normalization following Ramachandra & Mel 2013
    # Then Olshausen whitening.
    # size should be less than or equal to 21. (That's the size of the stored patches)
    patches = cPickle.load(open('/Users/gregv/Desktop/collabs/bartlett/data/o_patches.dat'))
    patches = patches[:, :size, :size]
    patches = patches.reshape((patches.shape[0], -1)) / np.max(patches)
    return patches[:-test_size], patches[-test_size:]


def bottom_half(x):
    """Obscure bottom half of images for inpainting experiments."""
    return np.hstack([x[:,:x.shape[1]/2], np.zeros((x.shape[0], x.shape[1]/2))])


def right_half(x):
    """Obscure right half of images for inpainting experiments."""
    if x.shape[1] == 784:
        #CHECK MODIFICATIONS FOR ORIGINAL_DIM = 784
        n1 = int(np.sqrt(x.shape[1]))
        n2 = n1
        half = 14  
    elif x.shape[1] == 2352:
        n1 = 28
        n2 = 84
        half = int(3*n/2)
    x = np.copy(x)
    x = x.reshape((-1, n1, n2))
    x[:, :, half:] = np.zeros((x.shape[0], n1, half))
    return x.reshape((-1, n1 * n2))


def flip(x):
    """Erase pixels with probability 1/2."""
    m = np.random.random(x.shape) < 0.75
    return np.where(m == 1, x, 1 - x)


def erase_chunk(x):
    x = np.copy(x)
    n = int(np.sqrt(x.shape[1]))
    n_samples = x.shape[0]
    x = x.reshape((-1, n, n))
    m_chunk = np.random.random((n_samples, n/4, n/4)) < 0.5  # Whether chunk is erased
    m = np.kron(m_chunk, np.ones((4, 4)))
    x = np.where(m == 1, x, 0)
    return x.reshape((-1, n * n))

def add_chunk(x):
    x = np.copy(x)
    n = int(np.sqrt(x.shape[1]))
    n_samples = x.shape[0]
    x = x.reshape((-1, n, n))
    m_chunk = np.random.random((n_samples, n/4, n/4)) < 0.5  # Whether chunk is erased
    m = np.kron(m_chunk, np.ones((4, 4)))
    x = np.where(m == 1, x, 0.5)
    return x.reshape((-1, n * n))


def erase_hstripe(x):
    x = np.copy(x)
    n = int(np.sqrt(x.shape[1]))
    n_samples = x.shape[0]
    x = x.reshape((-1, n, n))
    m_stripe = np.random.random((n_samples, n, 1)) < 0.5  # Whether chunk is erased
    m = np.repeat(m_stripe, n, axis=2)
    x = np.where(m == 1, x, 0)
    return x.reshape((-1, n * n))


def erase_vstripe(x):
    x = np.copy(x)
    n = int(np.sqrt(x.shape[1]))
    n_samples = x.shape[0]
    x = x.reshape((-1, n, n))
    m_stripe = np.random.random((n_samples, 1, n)) < 0.5  # Whether chunk is erased
    m = np.repeat(m_stripe, n, axis=1)
    x = np.where(m == 1, x, 0)
    return x.reshape((-1, n * n))

def add_hstripe(x):
    x = np.copy(x)
    n = int(np.sqrt(x.shape[1]))
    n_samples = x.shape[0]
    x = x.reshape((-1, n, n))
    m_stripe = np.random.random((n_samples, n, 1)) < 0.5  # Whether chunk is erased
    m = np.repeat(m_stripe, n, axis=2)
    x = np.where(m == 1, x, 0.5)
    return x.reshape((-1, n * n))

def add_vstripe(x):
    x = np.copy(x)
    n = int(np.sqrt(x.shape[1]))
    n_samples = x.shape[0]
    x = x.reshape((-1, n, n))
    m_stripe = np.random.random((n_samples, 1, n)) < 0.5  # Whether chunk is erased
    m = np.repeat(m_stripe, n, axis=1)
    x = np.where(m == 1, x, 0.5)
    return x.reshape((-1, n * n))

noises = [('no noise', lambda x: x), ('bottom half', bottom_half), ('right half', right_half), ('random flip', flip),
          ('-v stripe', erase_vstripe), ('-h stripe', erase_hstripe), ('erase chunk', erase_chunk),
          ('+v stripe', add_vstripe), ('+h stripe', add_hstripe),  ('add chunk', add_chunk)]


def plot_loss(hist, prefix=""):
    print "==> plotting loss function"
    plt.clf()
    plt.plot(hist.history['loss'], 'r', label='loss')
    plt.plot(hist.history['val_loss'], 'b',  label='val. loss')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('{}loss.png'.format(prefix))
    plt.close('all')
    with open('{}history.csv'.format(prefix), 'w') as f:
        for k in hist.history:
            f.write('{},{}\n'.format(k, ','.join(map(str, hist.history[k]))))


def vis_all_weights(model, prefix='', dataset = 'mnist', encoder_weights = False, label = ' '):
    print "==> visualize all weights"
    """Visualize encoder decoder weights at first layer."""
    w_dict = dict((w.name, w) for w in model.weights)
    #for k in w_dict:
            #if 'encoder1_W' in k:
            #    vis_weights(K.batch_get_value(w_dict[k]).T, '{}encoder'.format(prefix), dataset = dataset)
            #if hasattr(model.layers[1], 'get_w'):
            #    vis_weights(K.get_value(model.layers[1].get_w()).T, '{}encoder'.format(prefix), dataset = dataset)
    # for k in w_dict:
    #     if 'z_ci' in k:
    #         vis_weights(K.batch_get_value(w_dict[k]), '{}decoder'.format(prefix), dataset = dataset, label = label)
    #     elif 'z_activation_W' in k:
    #         vis_weights(K.batch_get_value(w_dict[k]), '{}decoder'.format(prefix), dataset = dataset, label = label)
    
    #'print 'decoding weights for ', model.layers[-1].name
    #vis_weights(K.get_value(model.layers[-1].get_w()), '{}decoder'.format(prefix), dataset = dataset, label = label)


        #    vis_weights(K.get_value(model.layers[-1].get_w()), '{}decoder'.format(prefix), dataset = dataset, label = label)
    if hasattr(model.layers[-1], 'get_w'):
        print 'decoding weights for ', model.layers[-1].name
        if not model.layers[-1].name.startswith('encoder'):
            vis_weights(K.get_value(model.layers[-1].get_w()), '{}decoder'.format(prefix), dataset = dataset, label = label)
        else:
            vis_weights(K.transpose(K.get_value(model.layers[-1].get_w())), '{}decoder'.format(prefix), dataset = dataset, label = label)
    elif hasattr(model.layers[-2], 'get_w'):
        #if not model.layers[-2].name.startswith('encoder'):
        print 'decoding weights for ', model.layers[-2].name
        if not model.layers[-2].name.startswith('encoder'):
            vis_weights(K.get_value(model.layers[-2].get_w()), '{}decoder'.format(prefix), dataset = dataset, label = label)
        else:
            vis_weights(K.transpose(K.get_value(model.layers[-2].get_w())), '{}decoder'.format(prefix), dataset = dataset, label = label)
    elif hasattr(model.layers[-3], 'get_w'):
        print 'decoding weights for ', model.layers[-3].name
        if not model.layers[-3].name.startswith('encoder'):
            vis_weights(K.get_value(model.layers[-3].get_w()), '{}decoder'.format(prefix), dataset = dataset, label = label)
        else:
            vis_weights(K.transpose(K.get_value(model.layers[-3].get_w())), '{}decoder'.format(prefix), dataset = dataset, label = label)
    # elif hasattr(model.layers[-4], 'get_w'):
    #     print 'decoding weights for ', model.layers[-4].name
    #     if not model.layers[-4].name.startswith('encoder'):
    #         vis_weights(K.get_value(model.layers[-4].get_w()), '{}decoder'.format(prefix), dataset = dataset, label = label)
    #     else:
    #         vis_weights(K.transpose(K.get_value(model.layers[-4].get_w())), '{}decoder'.format(prefix), dataset = dataset, label = label)
    else:
        print 'entering else'
        for k in w_dict:
            print k
            if 'decoder_W' in k:
                print 'decoding weights for decoder'
                vis_weights(K.batch_get_value(w_dict[k]), '{}decoder'.format(prefix), dataset = dataset, label = label)
            #if 'z_ci_W' in k:
            #    vis_weights(K.batch_get_value(w_dict[k]), '{}decoder'.format(prefix), dataset = dataset, label = label)
            #elif 'z_activation_W' in k:
            #    vis_weights(K.batch_get_value(w_dict[k]), '{}decoder'.format(prefix), dataset = dataset, label = label)
        


def vis_all_weights_old(model, prefix='', dataset = 'mnist', encoder_weights = False, label = ''):
    """Visualize encoder decoder weights at first layer."""
    w_dict = dict((w.name, w) for w in model.weights)
    #for k in w_dict:
            #if 'encoder1_W' in k:
            #    vis_weights(K.batch_get_value(w_dict[k]).T, '{}encoder'.format(prefix), dataset = dataset)
            #if hasattr(model.layers[1], 'get_w'):
            #    vis_weights(K.get_value(model.layers[1].get_w()).T, '{}encoder'.format(prefix), dataset = dataset)
    for k in w_dict:
        if 'decoder1_W' in k:
            vis_weights_old(K.batch_get_value(w_dict[k]), '{}decoder'.format(prefix), dataset = dataset, label = label)
    if hasattr(model.layers[-1], 'get_w'):
        print 'decoding weights for ', model.layers[-1].name
        vis_weights_old(K.get_value(model.layers[-1].get_w()), '{}decoder'.format(prefix), dataset = dataset, label = label)
    elif hasattr(model.layers[-2], 'get_w'):
        print 'decoding weights for ', model.layers[-2].name
        vis_weights_old(K.get_value(model.layers[-2].get_w()), '{}decoder'.format(prefix), dataset = dataset, label = label)
    elif hasattr(model.layers[-3], 'get_w'):
        print 'decoding weights for ', model.layers[-3].name
        vis_weights_old(K.get_value(model.layers[-3].get_w()), '{}decoder'.format(prefix), dataset = dataset, label = label)

def calc_score(data):
    #if not (data.shape[0] == 28 and data.shape[1] == 84):
    #    print 'please give 28x84 vector'
    overall_sum = 0
    char_sum = []
    for i in range(3):
        char_sum.append(np.sum(np.abs(data[:, 28*i:28*(i+1)]), axis= (0,1)))
        overall_sum = overall_sum + char_sum[i]  
    return np.divide(np.array(char_sum), overall_sum)


def vis_weights_old(w, prefix, dataset = '', label = ''):
    width = int(np.sqrt(w.shape[1]))
    if width * width != w.shape[1] or width > 100:
        print('wrong size weights:{} for visualization, assuming squares'.format(w.shape))
        return False
    m = w.shape[0]
    n = int(np.ceil(np.sqrt(m)))
    fig, ax = plt.subplots(n, n, figsize=(8, 8))
    for i in range(n):
        for j in range(n):
            if n * i + j < m:
                vmax = np.max(np.abs(w[i*n + j]))
                ax[i][j].imshow(w[i*n + j].reshape((width,width)), cmap=plt.cm.seismic, vmin=-vmax, vmax=vmax)
            ax[i][j].get_xaxis().set_visible(False)
            ax[i][j].get_yaxis().set_visible(False)
    plt.savefig('{}_weights.png'.format(prefix))
    plt.clf()

    # # Ng style plots
    # w /= np.sqrt(np.sum(w**2, axis=1, keepdims=True))
    # fig, ax = plt.subplots(n, n, figsize=(8, 8))
    # for i in range(n):
    #     for j in range(n):
    #         if n * i + j < m:
    #             ax[i][j].imshow(w[i*n + j].reshape((width,width)), cmap='gray')
    #         ax[i][j].get_xaxis().set_visible(False)
    #         ax[i][j].get_yaxis().set_visible(False)
    # plt.savefig('{}_weights_ng.png'.format(prefix))
    plt.close('all')
    return True

def vis_weights(w, prefix, dim = 12, dataset = 'mnist', label = ''):
    width = np.zeros((2,))
    if dataset == 'emnist' or dataset == 'emnist_seq':
        #print 'entering decoder reshape'
        width[0] = 28
        width[1] = 84
    elif dataset == 'mnist':
        width[0] = 28 #int(np.sqrt(w.shape[1]))
        width[1] = 28 #width[0]

    width[0] = int(width[0])
    width[1] = int(width[1])
    #w = np.array(w)
    #print 'width, weight shape: ', width, w.shape
    #if width[0] * width[1] != w.shape[1] or width[0] > 100 or width[1] > 100:
    #    print('wrong size weights:{} for visualization, assuming squares'.format(w.shape))
    #    return False

    if isinstance(w, np.ndarray): # if numpy array
        m = w.shape[0]
        m = int(m)
    else:
        print 'EVAL'
        m, a = w.get_shape()
        m = int(m)
        w = K.eval(w)
    n = int(m/4.)  #np.ceil(np.sqrt(m)))
    print 'w shape: ', w.shape
    print 'w ind: ', w[0].shape
    
    fig, axes = plt.subplots(int(np.sqrt(m)),int(m / int(np.sqrt(m))), sharex = True, sharey = True) 
    fig.set_figheight(4)
    fig.set_figwidth(9)
    scores = np.zeros((m, 3))
    print 'before for loop'
    for i in range(int(np.sqrt(m))):
        for j in range(int(m / int(np.sqrt(m)))):
            ind = int(int(m / int(np.sqrt(m)))*(i)+j)
            ind = int(ind)
            #print w[ind,:].shape, w[int(ind)].shape
            vmax = np.max(np.abs(w[ind]))
            #my_w = w[int(ind)]
            #my_w = my_w.reshape((int(width[0]),int(width[1])))
            imgplot = axes[int(i),int(j)].imshow(w[ind].reshape((int(width[0]),int(width[1]))), cmap=plt.cm.seismic, vmin=-vmax, vmax=vmax)
            # imgplot.set_interpolation('nearest')
            axes[i,j].tick_params(
                axis='both',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',
                left = 'off',         # ticks along the top edge are off
                labelbottom='off',
                labelleft = 'off') # labels along the bottom edge are off

            scores[ind] = calc_score(w[ind].reshape((int(width[0]),int(width[1]))))

    print 'scores, ', scores
    plt.suptitle(str(label + " Latent Factors"), fontsize=12, fontweight = 'heavy')
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0) 
    plt.subplots_adjust(top=0.99)
    fig.savefig('{}_weights_all.png'.format(prefix))

    for i in range(m):
        fig,ax = plt.subplots(1,1, figsize=(8, 8), squeeze=False)
        vmax = np.max(np.abs(w[i]))
        ax[0][0].imshow(w[i, :].reshape((int(width[0]),int(width[1]))), cmap=plt.cm.seismic, vmin=-vmax, vmax=vmax)
        #ax[i][j].get_xaxis().set_visible(False)
        #ax[i][j].get_yaxis().set_visible(False)

    #fig, ax = plt.subplots(3*n, n, figsize=(24, 24), squeeze=False)
    # for i in range(3*n): #rows
    #     for j in range(n):
    #         if n * i + j < m:
    #             vmax = np.max(np.abs(w[i*n + j]))
    #             ax[i][j].imshow(w[i*n + j].reshape((width[0],width[1])), cmap=plt.cm.seismic, vmin=-vmax, vmax=vmax)
    #         ax[i][j].get_xaxis().set_visible(False)
    #         ax[i][j].get_yaxis().set_visible(False)
        plt.savefig('{}_weights_{}.png'.format(prefix, str(i)))
        #plt.savefig('{}_weights.png'.format(prefix))
        plt.clf()

    # Ng style plots
    # w /= np.sqrt(np.sum(w**2, axis=1, keepdims=True))
    # fig, ax = plt.subplots(n, n, figsize=(8, 8))
    # for i in range(n):
    #     for j in range(n):
    #         if n * i + j < m:
    #             ax[i][j].imshow(w[i*n + j].reshape((width,width)), cmap='gray')
    #         ax[i][j].get_xaxis().set_visible(False)
    #         ax[i][j].get_yaxis().set_visible(False)
    # plt.savefig('{}_weights_ng.png'.format(prefix))
    plt.close('all')
    return True


def vis_reconstruction(model, data, prefix='', noise=None, n=8, dataset= 'mnist', merged = True, num_losses=1):
    print "==> visualizing reconstructions, prefix = {}".format(prefix)

    digit_size = [0,0]
    if dataset == 'mnist':
        # n is number of digits to reconstruct.
        digit_size[0] = int(np.sqrt(data.shape[1]))
        digit_size[1] = int(np.sqrt(data.shape[1]))
        print digit_size[0]
    elif dataset == 'emnist' or dataset == 'emnist_seq':
        # 28 x 84
        digit_size[0] = int(data.shape[1]/84)
        digit_size[1] = int(data.shape[1]/28)

    figure = np.ones((digit_size[0] * 3, digit_size[1] * n))
    
    print 'DATA SHAPE.... ', data.shape
    data_dim = data.shape[1]
    #if merged:
    #    dummy = Model(input = model.input, output = model.output[:-1, :data_dim])
    #    xbars = dummy.predict(data)
    
    if noise is None:
        inp = [data]*num_losses if num_losses > 1 else data
        xbars = model.predict(inp)
        
    else:
        data_noise = noise(data)
        inp = [data_noise]*num_losses if num_losses > 1 else data_noise
        xbars = model.predict(inp)
    
    if isinstance(xbars, list) and len(xbars) > 1:
        i = 0
        while xbars[i].shape[1] != digit_size[0]*digit_size[1]:
            i = i+1
        xbars = xbars[i]

    #xbars = xbars[data.shape[0], data.shape[1]]
    
    #print 'XBARS SHAPE **** ', xbars.shape
    #if xbars.shape[1] > data_dim:
    #    xbars = xbars[:-2,:datadim]
    #print data_dim, data[0].shape
    #for j in range(k):
    for i in range(n):
        digit = data[i].reshape(digit_size[0], digit_size[1])
        digit_decoded = xbars[i,:data_dim].reshape(digit_size[0], digit_size[1])
        figure[0 * digit_size[0]: (0 + 1) * digit_size[0],
               i * digit_size[1]: (i + 1) * digit_size[1]] = digit
        if noise is not None:
            figure[1 * digit_size[0]: (1 + 1) * digit_size[0],
                   i * digit_size[1]: (i + 1) * digit_size[1]] = data_noise[i].reshape((digit_size[0], digit_size[1]))
            figure[2 * digit_size[0]: (2 + 1) * digit_size[0],
                   i * digit_size[1]: (i + 1) * digit_size[1]] = digit_decoded
        else:
            figure[1 * digit_size[0]: (1 + 1) * digit_size[0],
                   i * digit_size[1]: (i + 1) * digit_size[1]] = digit_decoded
    plt.figure(figsize=(12, 24))
    plt.imshow(figure, cmap='Greys_r')
    plt.axis('off')
    plt.savefig('{}_reconstruction.png'.format(prefix), bbox_inches='tight')
    plt.close('all')

def vis_reconstruction_old(model, data, prefix='', noise=None, n=10, dataset= 'mnist'):
    if noise is None:
        xbars = model.predict(data)
    else:
        data_noise = noise(data)
        xbars = model.predict(noise(data))
    
    data_dim = data[0].shape[0]
    print data_dim
    for i in range(n):

        digit = data[i].reshape(digit_size, digit_size)
        digit_decoded = xbars[i][:data_dim].reshape(digit_size, digit_size)
        figure[0 * digit_size: (0 + 1) * digit_size,
               i * digit_size: (i + 1) * digit_size] = digit
        if noise is not None:
            figure[1 * digit_size: (1 + 1) * digit_size,
                   i * digit_size: (i + 1) * digit_size] = data_noise[i].reshape((digit_size, digit_size))
            figure[2 * digit_size: (2 + 1) * digit_size,
                   i * digit_size: (i + 1) * digit_size] = digit_decoded
        else:
            figure[1 * digit_size: (1 + 1) * digit_size,
                   i * digit_size: (i + 1) * digit_size] = digit_decoded

    plt.figure(figsize=(10, 4))
    plt.imshow(figure, cmap='Greys_r')
    plt.axis('off')
    plt.savefig('{}_reconstruction.png'.format(prefix), bbox_inches='tight')
    plt.close('all')    

def vis_classes(encoder, x_test, y_test, prefix='', batch_size=100):
    # display a 2D plot of the digit classes in the latent space, assuming it is 2d
    x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
    plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
    plt.colorbar()
    plt.savefig('{}classes.png'.format(prefix))

def vis_manifold(generator, prefix=''):
    # display a 2D manifold of the digits, assumes latent dimension has 2 vars.
    n = 15  # figure with 15x15 digits
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
    # to produce values of the latent variables z, since the prior of the latent space is Gaussian
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = generator.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig('{}manifold.png'.format(prefix))


def classify(x, y):
    clf = LinearSVC()
    scores = cross_val_score(clf, x, y, cv=10)
    print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std()))