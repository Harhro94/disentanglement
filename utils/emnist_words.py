import os
import sys
sys.path.insert(0, '../minsyn_ae')
import utilities
import numpy as np
import string
import random
from scipy import ndimage
from urllib import urlretrieve
import zipfile
import gzip
import shutil
import struct 
from pdb import set_trace as bp
import enchant

path = os.getcwd()

def get_top_words(num_words = 100, folder = path, top_words_path = 'top_words.txt'):
    text_file = open(os.path.join(folder, top_words_path))
    lines = text_file.read().split('\r')
    topwords = []
    for line in lines:
        topwords.append(line)
    return np.array(topwords[:num_words])


def read_data(dataset = "train",  letters_path = 'data', elements = 26):

    if dataset is "train":
        fname_img = os.path.join(path, letters_path, 'emnist-letters-train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, letters_path, 'emnist-letters-train-labels-idx1-ubyte')

    elif dataset is "test":
        fname_img = os.path.join(path, letters_path, 'emnist-letters-test-images-idx3-ubyte')
        fname_lbl = os.path.join(path, letters_path, 'emnist-letters-test-labels-idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    # Load everything in some numpy arrays
    try:
        with open(fname_lbl, 'rb') as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            lbl = np.fromfile(flbl, dtype=np.int8)
    except:
        try:
            os.mkdir(os.path.join(path, letters_path))
        except:
            pass
        try:
            os.mkdir(os.path.join(path, letters_path, zip_path))
        except:
            pass
        fn = os.path.join(path, letters_path, 'emnist.zip')
        if not os.path.isfile(fn):
            print 'downloading data set.  this may take some time!'
            url = urlretrieve('http://biometrics.nist.gov/cs_links/EMNIST/gzip.zip', fn)   
        print fn
        zip_path = 'gzip'
        with zipfile.ZipFile(fn, 'r') as zip_file:
            for member in zip_file.namelist():
                filename = os.path.basename(member)
                # skip directories
                if not filename:
                    continue
                if filename in ['emnist-letters-train-images-idx3-ubyte.gz', 'emnist-letters-test-images-idx3-ubyte.gz', 'emnist-letters-train-labels-idx1-ubyte.gz', 'emnist-letters-test-labels-idx1-ubyte.gz']:
                    zip_file.extract(member, os.path.join(path, letters_path))
                    f = gzip.open(os.path.join(path, letters_path, member), 'rb')
                    content = f.read()
                    f.close()
                    target = open(os.path.join(path, letters_path, os.path.splitext(filename)[0]), 'wb')
                    target.write(content)
                    target.close()
                    shutil.rmtree(os.path.join(path, letters_path, member), ignore_errors = True)
            zip_file.close()
        shutil.rmtree(os.path.join(path, letters_path, zip_path), ignore_errors = True)
        
        with open(fname_lbl, 'rb') as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            lbl = np.fromfile(flbl, dtype=np.int8)
   
    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)
   
    get_img = lambda idx: (lbl[idx], img[idx])


    letters = {x:[] for x in range(elements)}
    for i in xrange(len(lbl)):
        letters[lbl[i]-1].append(img[i])
    for i in range(elements):
        letters[i] = np.array(letters[i])#.reshape((len(letters[i]), square_dim))
    return letters


def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    vmax = np.max(np.abs(image))
    imgplot = ax.imshow(image, cmap=mpl.cm.seismic, vmin=-vmax, vmax=vmax)
    imgplot.set_interpolation('nearest')
    #ax.axis('off')
    #ax.xaxis.set_ticks_position('bottom')
    #ax.yaxis.set_ticks_position('right')
    pyplot.show()

def save(image, fn, save_path = 'examples'):
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    vmax = np.max(np.abs(image))
    imgplot = ax.imshow(image, cmap=mpl.cm.seismic, vmin=-vmax, vmax=vmax)
    imgplot.set_interpolation('nearest')
    #ax.axis('off')
    #ax.xaxis.set_ticks_position('bottom')
    #ax.yaxis.set_ticks_position('right')
    try:
        os.mkdir(os.path.join(path, save_path))
    except:
        pass
    fig.savefig(os.path.join(path, save_path, fn))
    pyplot.close('all')

def save_all(images, fn, save_path = 'examples', method = 'ICA'):
    from matplotlib import pyplot
    import matplotlib as mpl
    #pltsize = 2*len(images)
    #print len(images)
    fig, axes = pyplot.subplots(int(np.sqrt(len(images))), int(len(images))/int(np.sqrt(len(images))), sharex = True, sharey = True)
    fig.set_figheight(4)
    fig.set_figwidth(9)
    pyplot.subplots_adjust(wspace=0, hspace=0) 
    for i in range(int(np.abs(np.sqrt(len(images))))):
        for j in range(int(len(images))/int(np.sqrt(len(images)))):
            ind = int(len(images))/int(np.sqrt(len(images)))*(i)+j
            print ind
            print images[ind].shape
            vmax = np.max(np.abs(images[ind]))
            imgplot = axes[i, j].imshow(images[ind], cmap=mpl.cm.seismic, vmin=-vmax, vmax=vmax)
            # imgplot.set_interpolation('nearest')
            axes[i,j].tick_params(
                axis='both',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',
                left = 'off',         # ticks along the top edge are off
                labelbottom='off',
                labelleft = 'off') # labels along the bottom edge are off


    #ax.axis('off')
    #ax.xaxis.set_ticks_position('bottom')
    #ax.yaxis.set_ticks_position('right')
    try:
        os.mkdir(os.path.join(path, save_path))
    except:
        pass
    pyplot.suptitle(str(method + " Latent Factors"), fontsize=12, fontweight = 'heavy')
    fig.tight_layout()
    pyplot.subplots_adjust(wspace=0, hspace=0) 
    plt.subplots_adjust(top=0.99)
    fig.savefig(os.path.join(path, save_path, fn))
    pyplot.close('all')


def get_data(per_word = 500, seed = 0, data = 'train', num_words = 100, save_img = False, word_len = 3, addl_path = None, get_tw = True, sample = 'normal', set_letters = True):
    if addl_path is not None:
        path = addl_path
    else:
        path = os.getcwd() 
    if type(get_tw) is np.ndarray:
        print 'got array ', get_tw.shape
        top_words = get_tw
    elif get_tw:
        top_words = get_top_words(num_words = num_words, folder = path)
    else:
        print 'please enter numpy array or set get_top_words = True'
    np.random.seed(seed)
    letters_dict = read_data(dataset = data)
    original_dim = word_len*letters_dict[1][1].shape[0]*letters_dict[1][1].shape[1]
    if type(per_word) is np.ndarray:
        word_imgs = np.zeros((np.sum(per_word), original_dim))
        word_labels = [] #np.chararray((np.sum(per_word)))
    else:
        word_imgs = np.zeros((per_word*len(top_words), original_dim))
        word_labels = []#np.chararray((per_word*len(top_words)))
        #word_imgs = {x:[] for x n range(len(top_words))} 

    img_count = 0

    if type(set_letters) is np.ndarray:# and set_letters.shape[0] == 26:
        print 'using array for letters indices'
        letters_inds = set_letters
        set_letters = False
    else:
        letters_inds = np.zeros((26))

    for word_idx in range(len(top_words)):
        size = 1
        letter_idxs = []
        for lettr in top_words[word_idx]:
            letter_idx = ord(lettr.lower())-97
            letter_idxs.append(letter_idx)
            size = size * letters_dict[letter_idx].shape[0]
      

        if type(per_word) is np.ndarray:
            try:
                per_word_ = per_word[word_idx]
            except:
                print 'per_word array must have same length as top_words'
                return False
        else:
            per_word_ = per_word

        
        #        i = np.random.randint(0, len(letters_dict[letter_idxs[0]])-1)
        #        j = np.random.randint(0, len(letters_dict[letter_idxs[1]])-1)
        #        k = np.random.randint(0, len(letters_dict[letter_idxs[2]])-1)

        for m in range(per_word_):
        # TODO: same letter twice in word gets same sample of handwritten letter? or make both upper/lower case? 
            if set_letters:
                if m == 0 or not (sample == 'uniform'): #sample = 'normal' gives different letters within same word
                    i = np.random.randint(0, len(letters_dict[letter_idxs[0]])-1)
                    j = np.random.randint(0, len(letters_dict[letter_idxs[1]])-1)
                    k = np.random.randint(0, len(letters_dict[letter_idxs[2]])-1)

                if letters_inds[letter_idxs[0]] == 0:
                    letters_inds[letter_idxs[0]] = i
                if letters_inds[letter_idxs[1]] == 0: 
                    letters_inds[letter_idxs[1]] = j
                if letters_inds[letter_idxs[2]] == 0:
                    letters_inds[letter_idxs[2]] = k

            i = int(letters_inds[letter_idxs[0]])
            j = int(letters_inds[letter_idxs[1]]) 
            k = int(letters_inds[letter_idxs[2]])

            a = letters_dict[letter_idxs[0]]
            b = letters_dict[letter_idxs[1]]
            c = letters_dict[letter_idxs[2]]
            temp = np.concatenate((a[i],b[j]), axis = 0)
            temp = np.concatenate((temp, c[k]), axis = 0)

            word_imgs[img_count] = np.reshape(temp.T, (original_dim))
            word_labels.append(str(top_words[word_idx]))

            if save_img is True:
                fn = top_words[word_idx]+'_'+ str(m)+ '.pdf'
                #show(np.reshape(word_imgs[img_count], (temp.shape[1],temp.shape[0]))
                save(np.reshape(word_imgs[img_count], (temp.shape[1],temp.shape[0])), fn)

            img_count = img_count + 1
    #for word_idx in range(len(top_words)):
    print 'letters_inds: ', set_letters, ' : ', letters_inds
    return word_imgs, word_labels, letters_inds


def get_top_letter_words(letters = None):
    d = enchant.Dict("en_US")
    words = []
    test_words = []
    #letters = np.array(['e','o','a','t','n','y','s','r','u','d','i','l','w','h','b','g'])
    if letters == None:
        letters = {}
        letters[0] = ['a','o','b','l','h','s','t','c']#,'e','w','d','g','p','f','m','n']
        letters[1] = ['e','a','o','i','u','n','s','r']#,'l','w','g','f','t','c','d','y']
        letters[2] = ['t','e','y','n','r','d','w','o']#,'x','s','l','g','f','p','b','m']
    #old (by freq) ['t', 'a', 's', 'h','f','o','y','n','b','w','c','g','l','m','d', 'p']
    #old (by freq) ['h', 'n','o','a', 'e','u','i','l','s','w','r', 't','f','g','y', 'd']
    #old (by freq) ['e','d','t','r','y','u']#,'s','n','w','o','l','m','g','p','x', 'k']
    for i in letters[0]:
        for j in letters[1]:
            for k in letters[2]:
                if d.check(str(i+j+k)):
                    words.append(str(i+j+k))
                    print str(i+j+k)
                else:
                    #words.append(str(i+j+k))
                    test_words.append(str(i+j+k))
    return np.array(words), np.array(test_words)

def calc_score(data, square = False):
    if not (data.shape[0] == 28 and data.shape[1] == 84):
        print 'please give 28x84 vector'
    overall_sum = 0
    char_sum = []
    for i in range(3):
        if square:
            char_sum.append(np.sum(data[:, 28*i:28*(i+1)]**2, axis= (0,1)))
        else:
            char_sum.append(np.sum(np.abs(data[:, 28*i:28*(i+1)]), axis= (0,1)))
        overall_sum = overall_sum + char_sum[i]  
    return np.divide(np.array(char_sum), overall_sum)
