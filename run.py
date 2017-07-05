import sys
import minsyn_layers as ms
sys.path.insert(0, 'utils/')
import utilities
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import merge, Input, Dense, Dropout
from keras.layers import Activation, BatchNormalization, Lambda, Reshape
from keras.callbacks import Callback, TensorBoard
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras import objectives
from keras.layers.noise import GaussianNoise
from keras.callbacks import Callback, TensorBoard
from functools import partial, update_wrapper
import losses
import string
import models
from models import Args, EncoderArgs, DecoderArgs, SuperModel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--latent_dim', type=int, nargs='+', default=[12])
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--momentum', type=float, default=0.0, help="param for minsyn layer models")
parser.add_argument('--betas', type=float, nargs='+', default=[])
parser.add_argument('--sched', type=int, nargs='+', default=[], help="param for fit(data, optimizer)")
parser.add_argument('--dataset', type=str, default='emnist')
parser.add_argument('--strategy', type=str, default='fully_connected')
args = parser.parse_args()
print args

optimizer = Adam(lr=0.001, beta_1=0.5)


# IF USING EMNIST, DETERMINE WHICH LETTERS CAN APPEAR IN EACH POSITION
letters = {}
letters[0] = list(string.ascii_lowercase)
letters[1] = list(string.ascii_lowercase)
letters[2] = list(string.ascii_lowercase)  
# alternatively, top 16 most common in each position
#letters[0] = ['a','o','b','l','h','s','t','c','e','w','d','g','p','f','m','n']
#letters[1] = ['e','a','o','i','u','n','s','r','l','w','g','f','t','c','d','y']
#letters[2] = ['t','e','y','n','r','d','w','o','x','s','l','g','f','p','b','m']

# GET DATA
print "==> loading the dataset"
if args.dataset == 'emnist':
	x_train, x_test, y_train, y_test = utilities.get_emnist_lettercomb(letters)
elif args.dataset == 'mnist':
	x_train, x_test, y_train, y_test = utilities.mnist_data()

decoder_dim = list(reversed(args.latent_dim[:-1]))

f = Args(epochs = args.epochs, batch_size = args.batch_size, lagr_mult = args.betas,
		 anneal_sched = args.sched, optimizer = optimizer, momentum = args.momentum,
		 original_dim = x_train.shape[1])

"""
NOTE: Some models use objectives.binary_crossentropy, some use losses.error_entropy for recon
some losses: losses.error_entropy, objectives.binary_crossentropy,
			 objectives.mean_squared_error, losses.log_euclidean
"""
print "==> building the model"
super_model_strategy = args.strategy

if args.strategy == 'fully_connected':
	e = EncoderArgs(args.latent_dim, activation = 'softplus')
	d = DecoderArgs(decoder_dim, original_dim = x_train.shape[1], activation='sigmoid')
	#recon = objectives.binary_crossentropy
	recon = objectives.mean_squared_error
	#recon = losses.log_euclidean
	recon_weight = 1


if args.strategy == 'minsyn_decoder':
	e = EncoderArgs(args.latent_dim, activation = 'softplus', initializer = 'orthogonal')
	d = DecoderArgs(initializer = 'orthogonal', minsyn = 'binary', original_dim = x_train.shape[1])
	recon = objectives.binary_crossentropy
	recon_weight = 1


if args.strategy == 'info_dropout_ci_reg':
	args.betas = [0.0, 10**-5, 10**-4]
	args.sched = [0, 20, 80]
	e = EncoderArgs(args.latent_dim, info_dropout = True, ci_reg = True)
	d = DecoderArgs(decoder_dim, original_dim = x_train.shape[1])
	recon = objectives.binary_crossentropy
	recon_weight = 1


if args.strategy == 'ci_reg_enc':
	if dataset == 'mnist':
		args.betas = [10**-5, 10**-4]
		args.sched = [0, 10]
	else:
		args.betas = [0.0, 10**-5, 10**-4, 10**-3]
		args.sched = [0, 10, 50, 100]

	e = EncoderArgs(args.latent_dim, activation = 'softplus', minsyn = 'gaussian', ci_reg = True)
	d = DecoderArgs(decoder_dim, original_dim = x_train.shape[1])
	recon = objectives.binary_crossentropy
	recon_weight = 1


if args.strategy == 'ci_reg_dec':
	#[10**-5, 10**-4], [0, 8]
	args.betas = [10**-5, 10**-4, 10**-3]
	#betas = [10*x for x in betas]
	args.sched = [0, 20, 80]
	#e = EncoderArgs(args.latent_dim, info_dropout = True)
	e = EncoderArgs(args.latent_dim, activation = 'softplus', initializer = 'orthogonal')
	d = DecoderArgs(minsyn = 'binary', ci_reg = True, initializer = 'orthogonal', original_dim = x_train.shape[1])
	super_model_strategy = 'ci_reg_decoder'
	recon = objectives.binary_crossentropy
	recon_weight = 1


if args.strategy == 'screening':
	#args.betas = [0.25, .5, .9]
	#args.sched = [0, 10, 20]
	e = EncoderArgs(args.latent_dim, activation = 'softplus')
	d = DecoderArgs(screening = True, original_dim = x_train.shape[1], activation = 'sigmoid')
	#recon = losses.error_entropy
	recon = losses.log_euclidean
	#recon = objectives.mean_squared_error
	#recon_weight = 1
	recon_weight = 1 - args.betas[0]


if args.strategy == 'info_dropout':
	args.betas = [10**-5, 10**-3, 10**-2]
	args.sched = [0, 20, 80]

	e = EncoderArgs(args.latent_dim, info_dropout = True)
	d = DecoderArgs(original_dim = x_train.shape[1])
	super_model_strategy = 'dropout'
	recon = objectives.binary_crossentropy
	recon_weight = 1


if args.strategy == 'ci_wms':
	args.betas = 10**-4
	e = EncoderArgs(args.latent_dim, activation = 'softplus')
	d = DecoderArgs(minsyn='binary', ci_wms = True, original_dim = x_train.shape[1])
	recon = objectives.binary_crossentropy
	recon_weight = 1


model = SuperModel(strategy = super_model_strategy,
				   encoder = e,
				   decoder = d,
				   args = f,
				   recon = recon,
				   recon_weight = recon_weight)

model.fit(x_train, x_test)

