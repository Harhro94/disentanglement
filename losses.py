import sys
sys.path.insert(0, 'utils/')
import utilities
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import objectives
from keras.layers import Lambda, merge
from keras.callbacks import Callback, TensorBoard
import string
#from keras.activations import softmax

EPS = 10**-6

def binary_ce(merged = True): #cross ent + unmerge
	def my_loss(x, merged_decode):
		if not merged:
			x_decode = merged_decode
		else: #if shapes not identical, unmerge
			n = merged_decode.shape[1] / 2
			x_decode = merged_decode[:, :n]
			ci_decode = merged_decode[:, n:]
		return objectives.binary_crossentropy(x, x_decode)
	return my_loss

def binary_ci(x_true, merged_decode): #NOTE: x_true not used, here for easy use as Keras loss function
	#unpack merged_decode by default, use ci_tensor to enter ci dist separately
	batch, n = merged_decode.get_shape()
	n = int(n/2)
	x_decode = merged_decode[:, :n]
	ci_decode = merged_decode[:, n:]

	ci = -objectives.binary_crossentropy(x_decode, x_decode) + objectives.binary_crossentropy(x_decode, ci_decode)
	return n*K.mean(ci) # = sum over xi (objectives returns mean across x_i for batch)

def gaussian_ci(ci_R, log_noise = None): 
	def my_loss(x_true, merged_decode):
		batch, n = merged_decode.get_shape()
		n = int(n/2)
		x_decode = merged_decode[:, :n]
		ci_decode = merged_decode[:, n:]
		# prediction_i | y = Normal ( NN output , error var (***INSTEAD...running avg across batches?***))
		if log_noise is None: #if no (Gaussian) noise provided, x_true / variance of pred error used 
			#Note, this ideally would be over each x_i (variance taken within epoch?)
			#callback update variance values
			print 'Gaussian CI for Decoder.  If you intended for encoder, please feed a log_noise tensor as input'
			full_var = K.var(x_true - x_decode, axis = 0) 
		else: 
			full_var = K.exp(log_noise)
		ci_var = 1/(1+ci_R)
		# ci : full -> ci = x_decode -> ci_decode
		ci = .5*((full_var/ci_var) + K.mean(ci_decode - x_decode, axis = 0)**2 / ci_var - 1 + K.log(ci_var/full_var))
		return K.mean(ci) #NOTE: SHOULD BE SUM BUT TOO BIG
	return my_loss


#SHOULD BE OBSOLETE
def sigmoid_ci(x, merged_decode):
	# assumes both merged_decode are continuous (e.g. nn activations vs. gaussian z_ci).  approximate CI using binary formula
	batch, n = merged_decode.get_shape()
	n = int(n/2)
	x_decode = merged_decode[:, :n]
	ci_decode = merged_decode[:, n:]

	x_decode = K.sigmoid(x_decode)
	ci_decode = K.sigmoid(ci_decode)
	ci = -objectives.binary_crossentropy(x_decode, x_decode) + objectives.binary_crossentropy(x_decode, ci_decode)
	return n*K.mean(ci) # = sum over xi (objectives returns mean across x_i for batch)




def error_entropy(x_true, x_decode, invert_sigmoid = False, subtract_log_det=True, skew = False, kurt = False):
	H_NU = 0.5 * (1. + np.log(2 * np.pi))  # Entropy of standard gaussian
	# IF INVERT_SIGMOID = TRUE, x_true binary, invert sigmoid to do gaussian entropy estimation in domain R  
	if invert_sigmoid:
		# derivative of log (p / 1-p ) = 1 / p(1-p), log (this) = - log p(1-p) )
		# Do these cancel for screening? => subtract_log_det = false
		log_det_jac = -K.mean(K.log(K.clip(tf.multiply(x_true, 1-x_true), EPS, 1-EPS)))
		x_true = K.clip(x_true, EPS, 1-EPS)
		x_true = K.log(x_true) - K.log(1-x_true)
	else:
		log_det_jac = 0.0
	#log_det_jac = -K.mean(K.log(K.clip(tf.multiply(x_true, 1-x_true), EPS, 1-EPS))) if invert_sigmoid else 0 
	
	z = x_true - x_decode
	batch_mean = K.mean(z, axis=0)
	z = z - batch_mean
	batch_var = K.mean(z**2, axis=0)
	xi_var = K.var(x_true, axis = 0)
	out = -.5*K.sum(K.log(xi_var+EPS)) + .5*K.sum(1+ K.log(2*np.pi*(batch_var+EPS))) 
	out = out - log_det_jac if subtract_log_det else out
	if skew or kurt:
	    z = tf.div(z, K.sqrt(batch_var + EPS))  # TODO: tune. Avoids numerical errors, but just adding EPS leads to infinities (why?)
	if skew: #TODO: Check SCALE?
	    k1 = 36. / (8. * np.sqrt(3.) - 9.)
	    B = K.mean(tf.multiply(z, tf.exp(- 0.5 * tf.square(z))))
	    out -= k1 * K.sum(tf.square(B))
	if kurt: #TODO: Check SCALE?
	    k2 = 1. / (2. - 6. / np.pi)
	    k3 = np.sqrt(2. / np.pi)
	    A = K.mean(tf.abs(z))
	    out -= k2 * K.sum(tf.square(A - k3))
	return out


def log_euclidean(x_true, x_decode):
	""" Log euclidean loss function, used for screening
		loss = sum_i {log(E[(x_i - y_i) ** 2])}
	"""
	eps = 1e-2 # NOTE: this seem to be good, as the pixel precision in RGB is 10^-2 approximately
	errors = K.log(K.mean((x_true - x_decode) ** 2, axis=0) + eps)
	return 0.5 * K.sum(errors) # 0.5 is needed when pairing this loss with screening term


def info_dropout_kl(batch = 256, min_noise = False): #ONLY SOFTPLUS IMPLEMENTED
    def my_loss(x, merged_decode):
		b, n = merged_decode.get_shape()
		n = int(n/2)			
		z_output = merged_decode[:, n:]
		z_log_noise= merged_decode[:, :n]
    	
		# info dropout estimation of I(X:Z)
		mu_zj = K.mean(K.exp(z_output), axis=0)
		var_zj = K.var(K.exp(z_output), axis=0) + EPS
		alpha_2 = K.exp(z_log_noise) # variance of log_normal noise (mean 0).  alpha^2 in eq 5 of paper
		# Eq 5: Achille & Soatto : Information Dropout paper
		kl_loss = K.sum(.5/var_zj*(alpha_2 + mu_zj**2)-K.log(K.sqrt(alpha_2) / K.sqrt(var_zj)+EPS)-.5)
		
		return 1.0/batch*kl_loss if not min_noise else -1.0/batch*kl_loss
    return my_loss


def screening(n = 784, skew = False, kurt = False):
	print 'LOSS FUNCTION: Screening'

	def my_loss(x_true, merged_decode):
                x_decode = merged_decode[:, :n]
		z = merged_decode[:, n:]
		mi_ji = gaussian_mi(x_true, z)
		max_mi = K.max(mi_ji, axis=0)
		return -K.sum(max_mi)

        return my_loss


def ci_q_constraint(betas, n = 784):
	def my_loss(x_true, merged_decode):
		batch = K.cast(K.shape(x_true)[0], x_true.dtype)  # This is a node tensor, so we can't treat as integer
		div_N = Lambda(lambda v: v / batch_size)  
		
		# x_decode = p* = q(r|s)^beta
		x_decode = merged_decode[:, n:]
		z = merged_decode[:, :n]
		# q = output of CI beta layer = p(yj) prod p(xi | yj)
		#.5*((full_var/ci_var) + K.mean(ci_decode - x_decode, axis = 0)**2 / ci_var - 1 + K.log(ci_var/full_var))
		
		cond_var_p, mi_p, mj, vi_p, vj, V_p = gaussian_cond_var(x_true, z, return_all = True)
		cond_var_ps, mi_ps, mj, vi_ps, vj, V_ps = gaussian_cond_var(x_decode, z, return_all = True)
		
		#expand ALL to batch x j x i 
		z_ext = K.expand_dims(z, 2) 
		cond_mu_p = K.expand_dims(tf.divide(V_p, vj), 0) 
		cond_mu_ps = K.expand_dims(tf.divide(V_ps, vj), 0)
		mu_p_ext = K.expand_dims(x_true, 1)
		mu_ps_ext = K.expand_dims(x_decode, 1)
		cv_p_ext = K.expand_dims(cond_var_p, 0)
		cv_ps_ext = K.expand_dims(cond_var_ps, 0)
		mu_ij_p = K.expand_dims(mi_p, 0) + tf.multiply(z_ext, cond_mu_p) #batch x j x i
		mu_ij_ps = K.expand_dims(mi_ps, 0) + tf.multiply(z_ext, cond_mu_ps)



		log_q_xi_p = -.5*(1+K.log(2*np.pi)+ K.log(cond_var_p+EPS) + tf.divide((mu_p_ext - mu_ij_p)**2, cv_p_ext+EPS))
		log_q_xi_ps = -.5*(1+K.log(2*np.pi)+ K.log(cond_var_ps+EPS) + tf.divide((mu_ps_ext - mu_ij_ps)**2, cv_ps_ext+EPS))

		# log ( prod q(xi|yj) ) = log q(x|yj) = sum log q(xi|yj)
		q_expectation_p = K.mean(K.sum(log_q_xi_p, axis = -1), axis = 0)
		# mean is over batch for empirical estimate of expectation under p, p*
		q_expectation_ps = K.mean(K.sum(log_q_xi_ps, axis = -1), axis = 0)

		#betas = batch x j
		return K.abs(betas*(q_expectation_p-q_expectation_ps)) # 1 x j

	return my_loss
#def p_x_constraint:
#under x_decode or x_true?




def gaussian_mi(x, z): # returns j i matrix of I(X_i:Y_j)
	batch_size = K.cast(K.shape(x)[0], x.dtype)  # This is a node tensor, so we can't treat as integer
	div_N = Lambda(lambda v: v / batch_size)  

	mi = K.expand_dims(K.mean(x, axis=0), 0)  # mean of x_i
	mj = K.expand_dims(K.mean(z, axis=0), 1)  # mean of z_j
	vj = K.expand_dims(K.var(z, axis=0) + EPS, 1)  # sigma_j^2
	vi = K.expand_dims(K.var(x, axis=0) + EPS, 0)  # sigma_i^2
	V = div_N(K.dot(K.transpose(z-K.transpose(mj)), x- mi))
	rho = V / K.sqrt(vi*vj)
	return -.5*K.log(1-rho**2) #j i


def gaussian_cond_var(x_, z, invert_sigmoid = False, subtract_log_det = False, binary_z = False, return_all = False):
	batch_size = K.cast(K.shape(x_)[0], x_.dtype)  
	div_N = Lambda(lambda v: v / batch_size)  
	size = K.cast(K.shape(x_)[1], dtype='int32')

	if invert_sigmoid:
		#log_det_jac = -K.mean(K.log(K.clip(tf.multiply(x_, 1-x_), EPS, 1-EPS)))
		#log_det_jac = -K.mean(K.log(K.clip(tf.multiply(x, 1-x), EPS, 1-EPS)))
		x = K.clip(x_, EPS, 1-EPS)
		x = K.log(x) - K.log(1-x)
		if binary_z:
			z = K.clip(z, EPS, 1-EPS)
			z = K.log(z) - K.log(1-z)
			# z is likely continuous, but invert sigmoid if binary		
	else:
		x = x_
		log_det_jac = 0

	mi = K.expand_dims(K.mean(x, axis=0), 0)  # mean of x_i
	mj = K.expand_dims(K.mean(z, axis=0), 1)  # mean of z_j
	vj = K.expand_dims(K.var(z, axis=0), 1)  # sigma_j^2
	vi = K.expand_dims(K.var(x, axis=0), 0)  # sigma_i^2
	V = div_N(K.dot(K.transpose(z-K.transpose(mj)), x- mi)) #jxi
	cond_var = vi - tf.divide(V**2, vj)
	if return_all:
		return cond_var, mi, mj, vi, vj, V
	else:
		return cond_var

def gaussian_cond_ent(x_, z, invert_sigmoid = False, subtract_log_det = False, binary_z = False): # returns m x n matrix of I(Y_j:X_i)
	cond_var = gaussian_cond_var(x_, z, invert_sigmoid = False, binary_z = False)
	log_det_jac = -K.mean(K.log(K.clip(tf.multiply(x_, 1-x_), EPS, 1-EPS))) if invert_sigmoid else 0
	ent = .5*(1.+np.log(2 * np.pi) + K.log(cond_var+EPS)) 
	ent = ent - log_det_jac if subtract_log_det else ent
	return ent
	#return -.5*K.log(K.clip(1-rho**2, epsilon, 1-epsilon) )+ .5*(1.+np.log(2 * np.pi)+K.log(K.clip(vi, epsilon, 1-epsilon)))


def gaussian_ci_full(ci_R, log_noise = None):
	#D_KL[ p(yj|x) || p(yj) prod(xi|yj) / Z ]: log normal each sample || p(yj) log normal * prod(xi|yj) = log(ci decoder?) 
	#for encoder, nothing in x_true...
	def my_loss(x_true, merged_decode):
		# merged decode = z
		b, n = merged_decode.get_shape()
		n = int(n/2)			
		z_output = merged_decode[:, n:]
		z_log_noise= merged_decode[:, :n]
    	
		# info dropout estimation of I(X:Z)
		mu_zj = K.mean(K.exp(z_output), axis=0)
		var_zj = K.var(K.exp(z_output), axis=0) + EPS
		alpha_2 = K.exp(z_log_noise)

		# prediction_i | y = Normal ( NN output , error var (***INSTEAD...running avg across batches?***))
		if log_noise is None:
			full_var = K.var(x_true - x_decode, axis = 0) 
		else: 
			full_var = K.exp(log_noise)
		ci_var = 1/(1+ci_R)
		# ci : full -> ci = x_decode -> ci_decode
		ci = .5*((full_var/ci_var) + K.mean(ci_decode - x_decode, axis = 0)**2 / ci_var - 1 + K.log(ci_var/full_var))
		return K.mean(ci) #NOTE: SHOULD BE SUM BUT TOO BIG
	return my_loss


def ci_wms_dec(z_log_noise = None, batch = 256): # for use as regularization for CI decoder
	def my_loss(x, z):
		#batch, m = z.get_shape()
		#batch = K.cast(batch, x.dtype)
		#div_N = Lambda(lambda v: v / batch)
		
		z = z - K.mean(z, axis = 0, keepdims = True)
		x = x - K.mean(x, axis = 0, keepdims = True)
		x_2 = K.var(x, axis = 0, keepdims = True) #1xi
		z_2 = K.var(z, axis = 0, keepdims = True) #1xj
		#zj_xi = div_N(K.dot(K.transpose(z), x))	
		zj_xi = tf.divide(K.dot(K.transpose(z), x), batch)
		z2x2 = K.dot(K.transpose(z_2),x_2)


		R = 1./(1+K.sum(tf.divide(zj_xi**2, z2x2- zj_xi**2+EPS), axis = 0, keepdims = True)) #1 x i
		R = tf.divide(R, x_2 + EPS)
		x_decode = tf.multiply(R, K.transpose(K.dot(K.transpose(tf.divide(zj_xi, (z2x2 - zj_xi**2+EPS))), K.transpose(z))))
		
		dist = tf.contrib.distributions.Normal(0.0, 1.0)
		if z_log_noise is None:
			# for each yj, calculate prod (p(y_j) ) under gaussian
			prod_zj = K.sum(K.log(dist.prob(tf.divide(z, K.sqrt(z_2)))), axis = -1, keepdims = True) # batch x 1
			# why log?
		else:
			prod_zj = 0	
			# z_log_noise 1 x j
			# each real z, 

		# 1/n sum_y p_CI(xi|y) log p_CI(y) / prod(y_j)
		ci_wms = K.mean(tf.multiply(x_decode, K.log(K.dot(K.exp(prod_zj), R)+EPS)), axis = 0) #)), axis = 0) # 1 x i (mean over batch) 
		#ci_wms = K.mean(tf.multiply(x_decode, K.log(1./R)), axis = 0) #)), axis = 0) # 1 x i (mean over batch) 
		#return K.sum(tf.divide(zj_xi**2, z2x2-zj_xi**2+EPS))
		return K.sum(ci_wms) #we want min negative wms == minimize TC == minimize redundancy
	return my_loss
