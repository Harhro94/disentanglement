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
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--args.latent_dim', type=list, default=[12])
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--momentum', type=float, default=0.0, help="param for minsyn layer models")
parser.add_argument('--betas', type=list, default=[])
parser.add_argument('--sched', type=list, default=[], help="param for fit(data, optimizer)")
parser.add_argument('--dataset', type=str, default='emnist')
parser.add_argument('--strategy', type=str, default='fully_connected')
args = parser.parse_args()

optimizer = Adam(lr=0.0001)


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
if args.dataset == 'emnist':
	x_train, x_test, y_train, y_test = utilities.get_emnist_lettercomb(letters)
elif args.dataset == 'mnist':
	x_train, x_test, y_train, y_test = utilities.mnist_data()


for args.strategy in ['screening']:

	"""
	NOTE: Some models use objectives.binary_crossentropy, some use losses.error_entropy for recon
	"""
	if args.strategy == 'fully_connected':
		f = Args(epochs = args.n_epoch, batch_size = batch_size, lagr_mult = args.betas, anneal_sched = args.sched, 
					optimizer = optimizer, momentum = args.momentum, original_dim = x_train.shape[1])
		e = EncoderArgs(args.latent_dim, activation = 'softplus')
		d = DecoderArgs(list(reversed(args.latent_dim[:-1])))
		#losses.error_entropy, objectives.binary_crossentropy
		mymodel = SuperModel(strategy = args.strategy, encoder = e, decoder = d, args = f, recon =  objectives.binary_crossentropy,  recon_weight = 1)
		mymodel.fit(x_train, x_test)

	if args.strategy == 'minsyn_decoder':
		f = Args(epochs = args.n_epoch, batch_size = batch_size, lagr_mult = args.betas, anneal_sched = args.sched, 
					optimizer = optimizer, momentum = args.momentum, original_dim = x_train.shape[1])
		e = EncoderArgs(args.latent_dim, activation = 'softplus', initializer = 'orthogonal')
		d = DecoderArgs(initializer = 'orthogonal', minsyn = 'binary')
		mymodel = SuperModel(strategy = args.strategy, encoder = e, decoder = d, args = f, recon = objectives.binary_crossentropy,  recon_weight = 1)
		mymodel.fit(x_train, x_test)


	if args.strategy == 'info_dropout_ci_reg':
		args.betas = [0.0, 10**-5, 10**-4]
		args.sched = [0, 20, 80]
		f = Args(epochs = args.n_epoch, batch_size = batch_size, lagr_mult = args.betas, anneal_sched = args.sched, 
					optimizer = optimizer, momentum = args.momentum, original_dim = x_train.shape[1])
		e = EncoderArgs(args.latent_dim, info_dropout = True, ci_reg = True)
		d = DecoderArgs(list(reversed(args.latent_dim[:-1]))) #
		model = SuperModel(strategy = args.strategy, encoder = e, decoder = d, args = f, recon = objectives.binary_crossentropy, recon_weight = 1)
		model.fit(x_train, x_test)


	if args.strategy == 'ci_reg_enc':
		if dataset == 'mnist':
			args.betas = [10**-5, 10**-4]
			args.sched = [0, 10]
		else:
			args.betas = [0.0, 10**-5, 10**-4, 10**-3]
			args.sched = [0, 10, 50, 100]
		
		f = Args(epochs = args.n_epoch, batch_size = batch_size, lagr_mult = args.betas, anneal_sched = args.sched, 
						optimizer = optimizer, momentum = args.momentum, original_dim = x_train.shape[1])
		e = EncoderArgs(args.latent_dim, activation = 'softplus', minsyn = 'gaussian', ci_reg = True)
		d = DecoderArgs(list(reversed(args.latent_dim[:-1])))
		model = SuperModel(strategy = args.strategy, encoder = e, decoder = d, args = f, recon = objectives.binary_crossentropy, recon_weight = 1)
		model.fit(x_train, x_test)



	if args.strategy == 'ci_reg_dec':
		#[10**-5, 10**-4], [0, 8]
		args.betas = [10**-5, 10**-4, 10**-3]
		#betas = [10*x for x in betas]
		args.sched = [0, 20, 80]
		f = Args(epochs = args.n_epoch, batch_size = batch_size, lagr_mult = args.betas, anneal_sched = args.sched, 
						optimizer = optimizer, momentum = args.momentum, original_dim = x_train.shape[1])
		#e = EncoderArgs(args.latent_dim, info_dropout = True)
		e = EncoderArgs(args.latent_dim, activation = 'softplus', initializer = 'orthogonal')
		d = DecoderArgs(minsyn = 'binary', ci_reg = True, initializer = 'orthogonal')
		mimodel = SuperModel(strategy = 'ci_reg_decoder', encoder = e, decoder = d, args = f, recon = objectives.binary_crossentropy, recon_weight = 1)
		mimodel.fit(x_train, x_test)



	if args.strategy == 'screening':
		args.betas = [0.25, .5, .9]
		args.sched = [0, 10, 20]
		f = Args(epochs = args.n_epoch, batch_size = batch_size, lagr_mult = args.betas, anneal_sched = args.sched, 
					optimizer = optimizer, momentum = args.momentum, original_dim = x_train.shape[1])
		e = EncoderArgs(args.latent_dim, activation = 'softplus')
		d = DecoderArgs(screening_alpha = 10, screening = True)
		mymodel = SuperModel(strategy = args.strategy, encoder = e, decoder = d, args = f, recon = losses.error_entropy,  recon_weight = 1)
		mymodel.fit(x_train, x_test)



	if args.strategy == 'info_dropout':
		args.betas = [10**-5, 10**-3, 10**-2]
		args.sched = [0, 20, 80]

		f = Args(epochs = args.n_epoch, batch_size = batch_size, lagr_mult = args.betas, anneal_sched = args.sched, 
					optimizer = optimizer, momentum = args.momentum, original_dim = x_train.shape[1])
		e = EncoderArgs(args.latent_dim, info_dropout = True)
		d = DecoderArgs()
		mymodel1 = SuperModel(strategy = 'dropout', encoder = e, decoder = d, args = f, recon = objectives.binary_crossentropy, recon_weight = 1)
		mymodel1.fit(x_train, x_test)


	if args.strategy == 'ci_wms':
		args.betas = 10**-4
		f = Args(epochs = args.n_epoch, batch_size = batch_size, lagr_mult = args.betas, anneal_sched = args.sched, 
						optimizer = optimizer, momentum = args.momentum, original_dim = x_train.shape[1])
		e = EncoderArgs(args.latent_dim, activation = 'softplus')
		d = DecoderArgs(minsyn='binary', ci_wms = True)
		meamodel = SuperModel(strategy = args.strategy, encoder = e, decoder = d, args = f, recon = objectives.binary_crossentropy,  recon_weight = 1)
		meamodel.fit(x_train, x_test)

