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
#sys.argv = n_epoch, latent_dim, batch_size, momentum, betas, sched

dataset = 'emnist'

args = len(sys.argv)
n_epoch = int(sys.argv[1]) if args > 1 else 100
latent_dim = map(int, sys.argv[2].split(',')) if args > 2 else [12]
batch_size = int(sys.argv[3]) if args > 3 else 128
momentum = float(sys.argv[4]) if args > 4 else 0 #PARAM FOR MINSYN LAYER MODELS
betas = map(float, sys.argv[5].split(',')) if args > 5 else []
sched = map(int, sys.argv[6].split(',')) if args > 6 else []
 #PARAMETER FOR FIT(DATA, OPTIMIZER)

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
if dataset == 'emnist':
	x_train, x_test, y_train, y_test = utilities.get_emnist_lettercomb(letters)
elif dataset == 'mnist':
	x_train, x_test, y_train, y_test = utilities.mnist_data()


# INFO DROPOUT + ENCODER REG
run_regular = False
run_minsyn = True
run_dropout = False
run_dropout_minsyn = False
run_dropout_ci = False #True
run_ci_reg_enc = False
#run_ci_reg_enc = False
run_ci_reg_dec = True
run_screening = False
run_screening_mi = False
run_ci_wms = False

if run_dropout_ci:
	betas = [0.0, 10**-5, 10**-4]
	#betas = [x/x_train.shape[0] for x in betas]
	sched = [0, 20, 80]
	f = Args(epochs = n_epoch, batch_size = batch_size, lagr_mult = betas, anneal_sched = sched, 
				optimizer = optimizer, momentum = momentum, original_dim = x_train.shape[1])
	e = EncoderArgs(latent_dim, info_dropout = True, ci_reg = True)
	d = DecoderArgs(list(reversed(latent_dim[:-1])))
	meinmodel = SuperModel(strategy = 'dropout_ci', encoder = e, decoder = d, args = f, recon = objectives.binary_crossentropy, reg = [], recon_weight = 1)
	meinmodel.fit(x_train, x_test)


if run_regular:
	f = Args(epochs = n_epoch, batch_size = batch_size, lagr_mult = betas, anneal_sched = sched, 
				optimizer = optimizer, momentum = momentum, original_dim = x_train.shape[1])
	e = EncoderArgs(latent_dim, activation = 'softplus', initializer = 'glorot_uniform')
	d = DecoderArgs(list(reversed(latent_dim[:-1])), initializer = 'glorot_uniform')
	#losses.error_entropy, objectives.binary_crossentropy
	strategy = 'regular decoder'
	mymodel1 = SuperModel(strategy = strategy, encoder = e, decoder = d, args = f, recon =  objectives.binary_crossentropy, reg = [], recon_weight = 1)
	mymodel1.fit(x_train, x_test)

if run_minsyn:
	f = Args(epochs = n_epoch, batch_size = batch_size, lagr_mult = betas, anneal_sched = sched, 
				optimizer = optimizer, momentum = momentum, original_dim = x_train.shape[1])
	e = EncoderArgs(latent_dim, activation = 'softplus', initializer = 'orthogonal')
	d = DecoderArgs(initializer = 'orthogonal', minsyn = 'binary')
	#losses.error_entropy, objectives.binary_crossentropy
	strategy = 'minsyn'
	mymodel1 = SuperModel(strategy = strategy, encoder = e, decoder = d, args = f, recon = objectives.binary_crossentropy, reg = [], recon_weight = 1)
	mymodel1.fit(x_train, x_test)


if run_ci_reg_enc:
	#betas = [10**-3, 10**-2] # for ci binary
	if dataset == 'mnist':
		betas = [10**-5, 10**-4]
		#betas = [10*x for x in betas]
		sched = [0, 10]
	else:
		betas = [0.0, 10**-5, 10**-4, 10**-3]
		#betas = [10*x for x in betas]
		sched = [0, 10, 50, 100]
	
	f = Args(epochs = n_epoch, batch_size = batch_size, lagr_mult = betas, anneal_sched = sched, 
					optimizer = optimizer, momentum = momentum, original_dim = x_train.shape[1])
	#e = EncoderArgs(latent_dim, info_dropout = True)
	e = EncoderArgs(latent_dim, activation = 'softplus', minsyn = 'gaussian', ci_reg = True)
	d = DecoderArgs(list(reversed(latent_dim[:-1])))
	mimodel2 = SuperModel(strategy = 'ci_reg_encoder_beta', encoder = e, decoder = d, args = f, recon = objectives.binary_crossentropy, reg = [], recon_weight = 1)
	print len(betas), len(sched)
	mimodel2.fit(x_train, x_test)
	print len(betas), len(sched)



if run_ci_reg_dec:
	#[10**-5, 10**-4], [0, 8]
	betas = [10**-5, 10**-4, 10**-3]
	#betas = [10*x for x in betas]
	sched = [0, 20, 80]
	f = Args(epochs = n_epoch, batch_size = batch_size, lagr_mult = betas, anneal_sched = sched, 
					optimizer = optimizer, momentum = momentum, original_dim = x_train.shape[1])
	#e = EncoderArgs(latent_dim, info_dropout = True)
	e = EncoderArgs(latent_dim, activation = 'softplus', initializer = 'orthogonal')
	d = DecoderArgs(minsyn = 'binary', ci_reg = True, initializer = 'orthogonal')
	mimodel = SuperModel(strategy = 'ci_reg_decoder', encoder = e, decoder = d, args = f, recon = objectives.binary_crossentropy, reg = [], recon_weight = 1)
	mimodel.fit(x_train, x_test)


if run_dropout:
	#latent_dim[0] = latent_dim[0]-1
	betas = [10**-5, 10**-3, 10**-2]
	#betas = [x/x_train.shape[0] for x in betas]
	sched = [0, 20, 80]
	#f = Args(epochs = n_epoch, batch_size = batch_size, lagr_mult = betas, anneal_sched = sched, 
	#			optimizer = optimizer, momentum = momentum, original_dim = x_train.shape[1])
	#e = EncoderArgs(latent_dim, info_dropout = True)
	#d = DecoderArgs(minsyn = 'binary')
	#mymodel = SuperModel(encoder = e, decoder = d, args = f, recon = objectives.binary_crossentropy, reg = [], recon_weight = 1)
	#print mymodel.model.summary(s)
	#mymodel.fit(x_train, x_test)

	f = Args(epochs = n_epoch, batch_size = batch_size, lagr_mult = betas, anneal_sched = sched, 
				optimizer = optimizer, momentum = momentum, original_dim = x_train.shape[1])
	e = EncoderArgs(latent_dim, info_dropout = True)
	d = DecoderArgs()
	mymodel1 = SuperModel(strategy = 'dropout', encoder = e, decoder = d, args = f, recon = objectives.binary_crossentropy, reg = [], recon_weight = 1)
	mymodel1.fit(x_train, x_test)



if run_ci_wms:
	betas = 10**-4
	f = Args(epochs = n_epoch, batch_size = batch_size, lagr_mult = betas, anneal_sched = sched, 
					optimizer = optimizer, momentum = momentum, original_dim = x_train.shape[1])
	#e = EncoderArgs(latent_dim, info_dropout = True)
	e = EncoderArgs(latent_dim, activation = 'softplus')
	d = DecoderArgs(minsyn='binary', ci_wms = True)
	meamodel = SuperModel(strategy = 'ci_wms', encoder = e, decoder = d, args = f, recon = objectives.binary_crossentropy, reg = [], recon_weight = 1)
	meamodel.fit(x_train, x_test)

if run_screening_mi:
	# 4, 3 run prev (-4 before that, all 12)
	betas = [.5]
	sched = [0]
	f = Args(epochs = n_epoch, batch_size = batch_size, lagr_mult = betas, anneal_sched = sched, 
				optimizer = optimizer, momentum = momentum, original_dim = x_train.shape[1])
	e = EncoderArgs(latent_dim, activation = 'softplus')
	d = DecoderArgs(screening_alpha = 10, screening_mi = True)
	#losses.error_entropy, objectives.binary_crossentropy
	strategy = 'screening_ci_mi_test'
	mymodel = SuperModel(strategy = strategy,encoder = e, decoder = d, args = f, recon = losses.error_entropy, reg = [], recon_weight = 1)
	mymodel.fit(x_train, x_test)

if run_screening:
	# 4, 3 run prev (-4 before that, all 12)
	betas = [0.25, .5, .9]
	#betas = [.5, .75, .9]
	sched = [0, 10, 20]
	f = Args(epochs = n_epoch, batch_size = batch_size, lagr_mult = betas, anneal_sched = sched, 
				optimizer = optimizer, momentum = momentum, original_dim = x_train.shape[1])
	e = EncoderArgs(latent_dim, activation = 'softplus')
	d = DecoderArgs(screening_alpha = 10, screening = True)
	#losses.error_entropy, objectives.binary_crossentropy
	strategy = 'screening_ci_bin'
	mymodel = SuperModel(strategy = strategy,encoder = e, decoder = d, args = f, recon = losses.error_entropy, reg = [], recon_weight = 1)
	mymodel.fit(x_train, x_test)


