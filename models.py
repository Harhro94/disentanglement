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
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD
from keras import objectives
from keras import initializations
from keras.layers.noise import GaussianNoise
from keras.callbacks import Callback, TensorBoard
from functools import partial, update_wrapper
import losses
import string

#Basic Training Arguments:  epochs, batch, original_dim, lagrange multiplier schedules
class Args:
	def __init__(self, epochs = 100, batch_size = 256, original_dim = 784, lagr_mult = 1, anneal_sched = [], 
						optimizer = Adam(lr=0.0001), momentum = 0, shuffle = True):
		self.epochs = epochs
		self.batch = batch_size

		self.lagr_mult = lagr_mult
		if isinstance(lagr_mult, list):
			self.lagr = 1 if self.lagr_mult == [] else lagr_mult[0]
		elif isinstance(lagr_mult, dict):
			pass #treat once loss functions added
		else: # single value
			self.lagr = lagr_mult

		self.anneal_sched = False if anneal_sched == [] else anneal_sched
		if isinstance(lagr_mult, list) and self.anneal_sched and (len(self.anneal_sched) != len(self.lagr_mult)):
			raise ValueError('Lagrange Multipliers and Annealing Schedule lists must be same length')

		self.optimizer = optimizer
		self.momentum = momentum
		self.original_dim = original_dim
		self.shuffle = shuffle
		#self.n_samples = 0


"""
Encoder specific arguments:  
- Architecture, Strategy (minsyn, info_dropout, ci_reg)
- Layer activations and initializations 
"""
class EncoderArgs:
	"""
	ci_reg = True:  add regularization for NN network encoder divergence from CI (yj|x)
	minsyn = 'binary' or 'gaussian' (note: used in ci_reg... minsyn needs some encoding of y)
	info_dropout = True:  add multiplicative noise layer and KL error term (I(X:Y) joint regularization)
	ci_wms : WORK IN PROGRESS
	"""
	def __init__(self, latent_dim, minsyn = False, info_dropout = False, ci_reg = False, ci_q = False, add_noise = False,
				use_data_marginals = True, activation = 'softplus', initializer = 'glorot_uniform'):
					# gaussian = False, binary = False, 
					# minsyn = False, ci_reg = False, 
					# info_dropout = False,
					# recon = 'binary_ce', reg ='binary_ci', 
					# activation = 'softplus', initializer = 'glorot_uniform'):
					# infodropout_kl
					# recon = objectives.binary_crossentropy, reg = losses.binary_ci)
		self.latent_dim = [latent_dim] if isinstance(latent_dim, int) else latent_dim
		self.minsyn = minsyn
		self.info_dropout = info_dropout
		self.ci_reg = ci_reg
		self.ci_q = ci_q
		self.activation = activation
		# change if we need latent layer to be [0,1] rather than continuous
		self.final_activation = 'sigmoid' if (ci_reg and minsyn == 'binary') else self.activation
		self.initializer = initializer
		self.add_noise = add_noise
		self.use_data_marginals = use_data_marginals          
	
	def set_architecture(self):
		self.passed_layers = {}

		if self.info_dropout:
			z_enc = ms.dense_k(x, self.latent_dim[:-1], activation = 'softplus', initializer = self.initializer, leave_last = True)
			z_mean = Dense(self.latent_dim[-1], activation = self.final_activation, init = self.initializer, name='z_mean')(z_enc)
			# To do : tune initializer to start with low noise?
			z_log_noise = Dense(self.latent_dim[-1], activation = 'softplus', init = 'zero', name = 'z_log_noise')(z_mean)
			z = Lambda(ms.sample_info_dropout, name = 'z_activation')([z_mean, z_log_noise])
			info_dropout_loss = merge([z_log_noise, z], name = 'info_dropout', mode='concat', concat_axis=-1)
			self.passed_layers['info_dropout'] = info_dropout_loss
			self.passed_layers['z_log_noise'] = z_log_noise
		#elif self.encoder.minsyn:
		#NOTE: minsyn encoder doesn't really make sense without a way of encoding y (info dropout or dense) (i.e. no direct minsyn encoder)

		elif self.ci_reg and not (self.minsyn == 'binary'): #or self.encoder.gaussian_noise:
			# single layer noise f(z): CI = D_KL(p(yj|x)...) => need probability around gaussian mean f_j(x) (NN activation)
			z_enc = ms.dense_k(x, self.latent_dim[:-1], activation = self.activation, initializer = self.initializer, leave_last = True)
			z_mean = Dense(self.latent_dim[-1], activation= self.activation, name='z_m')(z_enc)
			z_log_noise = Dense(self.latent_dim[-1], activation = self.activation, init = 'zero', name='z_log_noise')(z_enc)
			if self.add_noise: 
					z = Lambda(ms.sample_vae, name='z_activation')([z_mean, z_log_noise])
			else:
					z = Lambda(ms.identity_layer, name='z_activation')(z_mean)
			self.passed_layers['z_log_noise'] = z_log_noise
		else:
			z = ms.dense_k(x, self.encoder.latent_dim, activation = self.encoder.final_activation, initializer = self.encoder.initializer)
		self.passed_layers['z_activation'] = z


		if self.ci_reg and self.use_data_marginals:
			merged_vector = merge([x, z], mode='concat', name = 'ci_q', concat_axis=-1)

			if self.minsyn == 'binary':  #ENSURE OUTPUT OF Z IS [0,1] (i.e. final_activation = sigmoid)
				z_ci = ms.DecodeB2B(self.encoder.latent_dim[-1], name = 'encoder_ci')(merged_vector)
			else:
				if self.encoder.ci_q:
					z_ci = ms.DecodeGaussianBeta(self.latent_dim[-1], name='encoder_ci')(merged_vector)
					bj = ms.DecodeGaussianBeta(self.latent_dim[-1], return_betas = True, name = 'b')(merged_vector)
					self.passed_layers['b'] = bj
				else: #compares with continuous encoder output
					z_ci = ms.DecodeGaussian(self.encoder.latent_dim[-1], name = 'encoder_ci')(merged_vector)
					# return r option makes output of minsyn layer = variances
				rj = ms.DecodeGaussian(self.encoder.latent_dim[-1], return_r = True, name = 'r')(merged_vector)
				self.passed_layers['r'] = rj
								
			self.passed_layers['encoder_ci'] = z_ci

			if self.ci_q:
					self.passed_layers['ci_q'] = merged_vector
											  
		return self.passed_layers
			


"""
Decoder specific arguments:  
- Architecture, Strategy (minsyn, info_dropout, ci_reg, screening)
- NOTE: latent_dim not required.  reconstruction layer added to ALL models, so only specify additional intermediate layers
- Layer activations and initializations 
- screening alpha for smooth max param
"""
class DecoderArgs:
	def __init__(self, latent_dim = [], minsyn = False, info_dropout = False, ci_reg = False, ci_wms = False, screening = False, screening_mi = False, 
					activation = 'softplus', initializer = 'glorot_uniform', screening_alpha = 1000, original_dim = 784):
		"""
		ci_reg = True:  add regularization for NN network decoder divergence from CI (xi|y)
		minsyn = 'binary' or 'gaussian' (please specify alongside ci_reg, or, ALONE = minsyn decoder)
		screening
		info_dropout = True: WORK IN PROGRESS add multiplicative noise to xi predictions 
		ci_wms : WORK IN PROGRESS
		"""

		self.latent_dim = latent_dim
		self.minsyn = minsyn
		self.info_dropout = info_dropout
		self.ci_reg = ci_reg
		self.activation = activation
		# change if we need latent layer to be [0,1] rather than continuous
		self.final_activation = self.activation
		self.initializer = initializer
		self.screening = screening
		self.alpha = screening_alpha
		self.ci_wms = ci_wms
		self.screening_mi = screening_mi
		self.original_dim = original_dim

	def set_architecture(self):
		#fully connected layers
		if len(self.latent_dim) > 0:
			self.latent_dim.append(self.original_dim)
			decoders = ms.dense_k(z, self.latent_dim[:-1], activation = self.activation, initializer = self.initializer, decoder = True)
			x_decode = Dense(self.original_dim, activation = self.final_activation, init = self.initializer, name ='decoder')(decoders)
		else:
			x_decode = Dense(self.original_dim, activation = self.final_activation, init = self.initializer, name ='decoder')(z)

		# DECODER OPTIONS
		if self.ci_reg or self.minsyn:
			merged_vector = merge([z, x], mode='concat', concat_axis=-1)
			if self.minsyn == 'gaussian':
				ci_decode = ms.DecodeGaussian(self.original_dim, name='decoder_ci')(merged_vector)
			else:
				ci_decode = ms.DecodeSigmoidFull(self.original_dim, name='decoder_ci')(merged_vector)
			

			if self.ci_reg: # x_decode (fully connected) is our decoder, loss function = recon + ci_reg
				merged_decode = merge([x_decode, ci_decode], mode = 'concat', name ='ci_decoder_reg', concat_axis = -1)

				if self.minsyn == 'gaussian':
					# return r option makes output of minsyn layer = variances for ci decoder
					ri = ms.DecodeGaussian(self.latent_dim[-1], return_r = True, name = 'r')(merged_vector)
					self.passed_layers['r'] = ri

				self.passed_layers['ci_decoder_reg'] = merged_decode
				
			else: # minsyn decoder only
				self.passed_layers['decoder_ci'] = ci_decode
			
		elif self.screening:			
			if self.minsyn and not self.ci_reg: # minsyn decoder
				merged_screen = merge([ci_decode, z], mode = 'concat', name = 'screening', concat_axis = -1)
			else: # full decoder
				merged_screen = merge([x_decode, z], mode = 'concat', name = 'screening', concat_axis = -1)
				self.passed_layers['screening'] = merged_screen
						
		if self.info_dropout: #UNTESTED: info dropout for learning noise to add to reconstruction
			# To do : initializer to start with low noise?
			xi_log_noise = Dense(self.original_dim, activation = 'softplus', init = 'zero', name = 'xi_log_noise')(x_decode if (not self.minsyn or self.ci_reg) else ci_decode)
			xi_out = Lambda(ms.sample_info_dropout, name = 'decoder_id')([x_decode if (not self.minsyn or self.ci_reg) else ci_decode, z_log_noise])
			info_dropout_loss = merge([xi_log_noise, xi_out], name = 'info_dropout', mode='concat', concat_axis=-1)
			self.passed_layers['info_dropout'] = info_dropout_loss
			self.passed_layers['decode_id'] = xi_out #for easy identification as decoder

		if (self.minsyn and not self.ci_reg) and not self.info_dropout:
				self.passed_layers['decoder'] = x_decode

				
class SuperModel:
	"""
		Initially meant to be a Super class, but now sets model architecture according to Encoder / Decoder args
		fit() according to Args according to annealing schedule		
		transform() to get latent from data, predict() to reconstruct


		Arguments:
		EncoderArgs, DecoderArgs, (fit) Args
		strategy: used for naming output folder location
		recon: reconstruction loss function (and its weight, default 1*error) 

		Constructor sets architecture and prints model summary
	"""

	def __init__(self, encoder, decoder, args, strategy = 'default', recon = objectives.binary_crossentropy, recon_weight = 1):

		self.encoder = encoder
		self.decoder = decoder
		self.args = args
		self.strategy = strategy
		self.recon = [recon] if isinstance(recon, str) else recon
		
		
		#necessary?
		self.num_losses = 0
		self.num_losses = self.num_losses+1 if self.encoder.info_dropout is True else self.num_losses
		self.num_losses = self.num_losses+1 if self.decoder.info_dropout is True else self.num_losses
		self.num_losses = self.num_losses+1 if self.encoder.ci_reg is True else self.num_losses
		self.num_losses = self.num_losses+1 if self.decoder.ci_reg is True else self.num_losses
		self.num_losses = self.num_losses+1 if self.encoder.ci_wms is True else self.num_losses
		self.num_losses = self.num_losses+1 if self.decoder.ci_wms is True else self.num_losses
		self.num_losses = self.num_losses+1 if self.decoder.screening is True else self.num_losses

		
		self.loss_function = {}
		self.loss_weights = [1]*self.num_losses

		self.inputs = []
		self.outputs = []
		self.recon_weight = recon_weight

		self.prepare_model()
		#self.set_architecture()
		print self.model.summary()

	def prepare_model(self):
		x = Input(shape=(self.args.original_dim,)) 
		
		self.inputs = []
		inputs = self.num_losses+1 if self.num_losses == 0 else self.num_losses
		for i in range(1): #inputs):
			self.inputs.append(x)
												  
		self.encoder_layers = self.encoder.set_architecture()
		self.decoder_layers = self.decoder.set_architecture()                                                  
		self.set_losses()

		self.num_losses = len(self.loss_function)
		if self.num_losses != len(self.outputs):
			print '**** LOSS FUNCTION AND OUTPUTS NOT SAME LENGTH ****'
		
		self.model = Model(input = self.inputs[0], output = self.outputs)
																  
	def set_losses(self):
		# RECON (+ info dropout)
		if self.decoder.minsyn and not self.decoder.ci_reg:
				self.losses['decoder_ci'] = self.recon
				self.outputs.append(self.decoder_layers['decoder_ci'])

		elif self.decoder.info_dropout:
				self.losses['decoder_id'] = self.recon
				self.outputs.append(self.decoder_layers['decoder_id'])
				self.losses['info_dropout'] = losses.info_dropout_kl(self.args.batch, min_noise = True)
				self.outputs.append(self.decoder_layers['info_dropout'])

		else:
				self.losses['decoder'] = self.recon
				self.outputs.append(self.decoder_layers['decoder'])

		# DECODER REGULARIZATION
		if self.decoder.ci_reg:
				# default to binary_ci with unspecified minsyn
				self.losses['ci_decoder_reg'] = losses.binary_ci if not self.decoder.minsyn == 'gaussian' else losses.gaussian_ci(self.decoder_layers['r'], self.encoder_layers['z_log_noise'])
				self.outputs.append(self.decoder_layers['ci_decoder_reg'])
		if self.decoder.ci_wms:
				self.losses['z_activation'] = losses.ci_dec_wms(batch = self.args.batch)
		if self.decoder.ci_q:
				raise NotImplementedError
		if self.decoder.screening:
				self.loss_function['screening'] = losses.screening(self.args.original_dim, self.decoder.alpha) 
				self.outputs.append(self.decoder_layers['screening'])

		# ENCODER REGULARIZATION
		if self.encoder.ci_reg:
			if not self.encoder.use_data_marginals:
				merged_vector = merge([self.decoder_layers['decoder'], self.encoder_layers['z_activation']], mode='concat', name = 'ci_q', concat_axis=-1)

				if self.minsyn == 'binary':  #ENSURE OUTPUT OF Z IS [0,1] (i.e. final_activation = sigmoid)
						z_ci = ms.DecodeB2B(self.encoder.latent_dim[-1], name = 'encoder_ci', momentum = self.args.momentum)(merged_vector)
				else:
					if self.encoder.ci_q:
						z_ci = ms.DecodeGaussianBeta(self.latent_dim[-1], name='encoder_ci')(merged_vector)
						bj = ms.DecodeGaussianBeta(self.latent_dim[-1], return_betas = True, name = 'b')(merged_vector)
						self.encoder_layers['b'] = bj
					else: #compares with continuous encoder output
						z_ci = ms.DecodeGaussian(self.encoder.latent_dim[-1], name = 'encoder_ci', momentum = self.args.momentum)(merged_vector)
					# return r option makes output of minsyn layer = variances
					rj = ms.DecodeGaussian(self.encoder.latent_dim[-1], return_r = True, name = 'r', momentum = self.args.momentum)(merged_vector)
					self.encoder_layers['r'] = rj
				self.encoder_layers['ci_q'] = merged_vector

			encoder_ci_loss = merge([self.encoder_layers['z_activation'], self.encoder_layers['z_ci']], mode = 'concat', name ='ci_encoder_reg', concat_axis=-1)
			self.outputs.append(encoder_ci_loss)

			if self.encoder.minsyn == 'gaussian':
				self.loss_function['ci_encoder_reg'] = losses.gaussian_ci(self.encoder_layers['r'], self.encoder_layers['z_log_noise'])                              
			else:
				self.loss_function['ci_encoder_reg'] = losses.binary_ci

			if self.encoder.ci_q:
				self.loss_function['ci_q'] = losses.ci_q_constraint(self.encoder_layers['b'], self.encoder.latent_dim[-1])
				self.outputs.append(self.encoder_layers['ci_q'])


		if self.encoder.info_dropout:
				self.loss_function['info_dropout'] = losses.info_dropout_kl(self.args.batch)
				self.outputs.append(self.encoder_layers['info_dropout'])
														  
	
	def set_architecture(self):
		"""
			Creates layers and adds loss functions based on options specified
		"""
		''' If adding layers, be sure to set layer names appropriately to match loss function dictionary
			z_activation for latent variables
			decoder for output layer (for reconstruction loss) '''

		x = Input(shape=(self.args.original_dim,)) 
		
		self.inputs = []
		inputs = self.num_losses+1 if self.num_losses == 0 else self.num_losses
		for i in range(1): #inputs):
			self.inputs.append(x)
		
		#ENCODER ARCHITECTURES
		if self.encoder.info_dropout:
			z_enc = ms.dense_k(x, self.encoder.latent_dim[:-1], activation = 'softplus', initializer = self.encoder.initializer, leave_last = True)
			z_mean = Dense(self.encoder.latent_dim[-1], activation = self.encoder.final_activation, init = self.encoder.initializer, name='z_mean')(z_enc)
			# To do : tune initializer to start with low noise?
			z_log_noise = Dense(self.encoder.latent_dim[-1], activation = 'softplus', init = 'zero', name = 'z_log_noise')(z_mean)
			z = Lambda(ms.sample_info_dropout, name = 'z_activation')([z_mean, z_log_noise])
			info_dropout_loss = merge([z_log_noise, z], name = 'info_dropout', mode='concat', concat_axis=-1)
		
		#elif self.encoder.minsyn:
		#NOTE: minsyn encoder doesn't really make sense without a way of encoding y (info dropout or dense) (i.e. no direct minsyn encoder)
		
		elif self.encoder.ci_reg: #or self.encoder.gaussian_noise:
			# noise added via VAE reparameterization trick:  CI = D_KL(p(yj|x)...) => need probability around gaussian mean f_j(x) (NN activation)
			z_enc = ms.dense_k(x, self.encoder.latent_dim[:-1], activation = self.encoder.activation, initializer = self.encoder.initializer, leave_last = True)
			z_mean = Dense(self.encoder.latent_dim[-1], activation= self.encoder.activation, name='z_m')(z_enc)
			z_log_noise = Dense(self.encoder.latent_dim[-1], activation = self.encoder.activation, init = 'zero', name='z_log_noise')(z_enc)
			z = Lambda(ms.sample_vae, name='z_activation')([z_mean, z_log_noise])
		else:
			z = ms.dense_k(x, self.encoder.latent_dim, activation = self.encoder.final_activation, initializer = self.encoder.initializer)
		
		
		# DECODER ARCHITECTURE:  fully connected layers
		if len(self.decoder.latent_dim) > 0:
			if self.decoder.latent_dim[-1] != self.args.original_dim:
				self.decoder.latent_dim.append(self.args.original_dim)
			decoders = ms.dense_k(z, self.decoder.latent_dim[:-1], activation = self.decoder.activation, initializer = self.decoder.initializer, decoder = True)
			x_decode = Dense(self.args.original_dim, activation = self.decoder.final_activation, init = self.decoder.initializer, name ='decoder')(decoders)
		else:
			x_decode = Dense(self.args.original_dim, activation = self.decoder.final_activation, init = self.decoder.initializer, name ='decoder')(z)

		# DECODER OPTIONS & LOSS FUNCTIONS
		if self.decoder.ci_reg or self.decoder.minsyn:

			merged_vector = merge([z, x], mode='concat', concat_axis=-1)
			if self.decoder.minsyn == 'gaussian':
				ci_decode = ms.DecodeGaussian(self.args.original_dim, name='decoder_ci', momentum = self.args.momentum)(merged_vector)
			else:
				ci_decode = ms.DecodeSigmoidFull(self.args.original_dim, name='decoder_ci', momentum = self.args.momentum)(merged_vector)
			

			if self.decoder.ci_reg: # x_decode (fully connected) is our decoder, loss function = recon + ci_reg
				merged_decode = merge([x_decode, ci_decode], mode = 'concat', name ='ci_decoder_reg', concat_axis = -1)
				self.loss_function['decoder'] = self.recon
				self.outputs.append(x_decode)

				if self.decoder.minsyn == 'gaussian':
					# return r option makes output of minsyn layer = variances
					ri = ms.DecodeGaussian(self.encoder.latent_dim[-1], return_r = True, name = 'r', momentum = self.args.momentum)(merged_vector)
					self.loss_function['ci_decoder_reg'] = losses.gaussian_ci(ri, z_log_noise)
				else:
					self.loss_function['ci_decoder_reg'] = losses.binary_ci
				self.outputs.append(merged_decode)
			else: # MINSYN decoder
				self.loss_function['decoder_ci'] = self.recon
				self.outputs.append(ci_decode)
			
				if self.decoder.ci_wms :
					self.loss_function['z_activation'] = losses.ci_wms_dec(batch = self.args.batch)
					self.outputs.append(z)


		elif self.decoder.screening:			
			if self.decoder.minsyn and not self.decoder.ci_reg: # minsyn decoder
				merged_screen = merge([ci_decode, z], mode = 'concat', name = 'screening', concat_axis = -1)
			else: # full decoder
				merged_screen = merge([x_decode, z], mode = 'concat', name = 'screening', concat_axis = -1)
			self.loss_function['decoder'] = self.recon
			self.outputs.append(x_decode)
			self.loss_function['screening'] = losses.screening(self.args.original_dim, self.decoder.alpha) 
			self.outputs.append(merged_screen)

		else:
			self.loss_function['decoder'] = self.recon
			self.outputs.append(x_decode)

		if self.decoder.info_dropout: #UNTESTED: info dropout for learning noise to add to reconstruction
			# To do : initializer to start with low noise?
			xi_log_noise = Dense(self.encoder.latent_dim[-1], activation = 'softplus', init = 'zero', name = 'xi_log_noise')(x_decode if (not self.decoder.minsyn or self.decoder.ci_reg) else ci_decode)
			xi_out = Lambda(ms.sample_info_dropout, name = 'z_activation')([x_decode if (not self.decoder.minsyn or self.decoder.ci_reg) else ci_decode, z_log_noise])
			
			info_dropout_loss = merge([xi_log_noise, xi_out], name = 'info_dropout', mode='concat', concat_axis=-1)
			
			self.loss_function['info_dropout'] = losses.info_dropout_kl(self.args.batch)
			self.outputs.append(info_dropout_loss)



		# ADD LOSS CONTRIBUTIONS INVOLVING ENCODER (now that decoder architecture set)
		if self.encoder.info_dropout:
			self.loss_function['info_dropout'] = losses.info_dropout_kl(self.args.batch)
			self.outputs.append(info_dropout_loss)
		
		if self.encoder.ci_reg:
			merged_vector = merge([x, z], mode='concat', name = 'merged', concat_axis=-1)
			
			if minsyn == 'binary':  #ENSURE OUTPUT OF Z IS [0,1] (i.e. final_activation = sigmoid)
				z_ci = ms.DecodeSigmoidFull(self.encoder.latent_dim[-1], name = 'encoder_ci', momentum = self.args.momentum)(merged_vector)
			else: #compares with continuous encoder output
				z_ci = ms.DecodeGaussian(self.encoder.latent_dim[-1], name = 'encoder_ci', momentum = self.args.momentum)(merged_vector)
				# return r option makes output of minsyn layer = variances
				rj = ms.DecodeGaussian(self.encoder.latent_dim[-1], return_r = True, name = 'r', momentum = self.args.momentum)(merged_vector)
			
			encoder_ci_loss = merge([z, z_ci], mode = 'concat', name ='ci_encoder_reg', concat_axis=-1)
			self.outputs.append(encoder_ci_loss)
			#replace with gaussian estimate
			self.loss_function['ci_encoder_reg'] = losses.gaussian_ci(rj, z_log_noise) #losses.sigmoid_ci()

		self.num_losses = len(self.loss_function)
		if self.num_losses != len(self.outputs):
			print '**** LOSS FUNCTION AND OUTPUTS NOT SAME LENGTH ****'
		

		self.model = Model(input = self.inputs[0], output = self.outputs)

		

	
	def fit(self, x_train, x_val): #note: x_val must have proper # of inputs in list
		def update_loss_weights():	# all regularizers have to have same weight.  
			# TO DO: allow matrix of annealing schedule (vs. list) for multiple regularizers 
			self.loss_weights = [1]*len(self.loss_function)
			self.loss_weights = [self.args.lagr*x for x in self.loss_weights]
			if self.decoder.screening or self.decoder.screening_mi:
				# subtract lagr if screening loss function doesn't include h(xi|z) estimation
				self.loss_weights[0] = self.recon_weight #- self.args.lagr
			else:
				self.loss_weights[0] = self.recon_weight
			print self.loss_weights
			
		
		self.x_train_list= []
		self.x_val_list= []
		for i in range(len(self.loss_function)):
			self.x_train_list.append(x_train)
			self.x_val_list.append(x_val)
		self.x_train = x_train
		self.x_val = x_val
		

		if not self.args.anneal_sched: # Train for full # epochs on same settings
			
			update_loss_weights()
			self.model.compile(optimizer = self.args.optimizer, loss = self.loss_function, loss_weights = self.loss_weights)

			# TO DO: Add callbacks
			self.model.fit(self.x_train, self.x_train_list, shuffle = self.args.shuffle, 
						nb_epoch= self.args.epochs, batch_size = self.args.batch, 
						validation_data = (self.x_val, self.x_val_list))#,
						#callbacks = [callbacks])

		else: # anneal according to lagr_mult, anneal_sched

			self.args.anneal_sched.append(self.args.epochs)
			for k in range(len(self.args.lagr_mult)):
				self.args.lagr = self.args.lagr_mult[k]
				n_epochs = int(min(self.args.anneal_sched[k+1], self.args.epochs)-self.args.anneal_sched[k])
				
				if n_epochs > 0:
					update_loss_weights()
					self.model.compile(optimizer = self.args.optimizer, loss = self.loss_function, loss_weights = self.loss_weights) 
					#self.model.compile(optimizer = optimizer, loss = self.loss_function(lagr = self.args.lagr) ) 
					# MAKE SURE TO ALWAYS WRITE loss_function(lagr)

					self.model.fit(self.x_train, self.x_train_list, shuffle = self.args.shuffle,
						nb_epoch = n_epochs,
						batch_size = self.args.batch, 
						validation_data = (self.x_val, self.x_val_list))#,
						#callbacks = [callbacks])
		
		self.visualize_all()

		#if self.encoder.ci_reg_beta:
		#	print self.model.get_layer('encoder_ci').get_betas().eval(session = K.get_session())
		#if self.decoder.ci_reg_beta:
		#	print self.model.get_layer('decoder_ci').get_betas().eval(session = K.get_session())


	def visualize_all(self):
		self.save_path()
		if self.args.original_dim == 784:
			dataset = 'mnist'
		elif self.args.original_dim == 2352:
			dataset = 'emnist'
		print dataset
		utilities.vis_all_weights(self.model, prefix=self.location, dataset = dataset, label = '')
		indices = np.random.randint(0, self.x_train.shape[0]-1, self.args.batch)
		utilities.vis_reconstruction(self.model, self.x_train[indices, :], prefix='{}train'.format(self.location), dataset = dataset, num_losses = 1) #self.num_losses)
		indices = np.random.randint(0, self.x_val.shape[0]-1, self.args.batch)
		utilities.vis_reconstruction(self.model, self.x_val[indices, :], prefix='{}test'.format(self.location), dataset = dataset, num_losses = 1)#self.num_losses)


	def set_lagr_mult(self, lagr_mult, anneal_sched = []):
		self.args.lagr = lagr_mult[0] if len(lagr_mult) == 1 else lagr_mult
		self.args.anneal_sched = False if anneal_sched == [] else anneal_sched


	def save_path(self):
		self.location = 'results/{}_{}_{}-{}/'.format(self.encoder.latent_dim, self.strategy, self.args.epochs, self.args.batch)
		utilities.make_sure_path_exists(self.location)
