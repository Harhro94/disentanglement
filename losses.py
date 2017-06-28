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


def gaussian_mi(x, z): # returns j i matrix of I(X_i:Y_j)
	batch_size = K.cast(K.shape(x)[0], x.dtype)  # This is a node tensor, so we can't treat as integer
	div_n = Lambda(lambda v: v / batch_size)  

	mi = K.expand_dims(K.mean(x, axis=0), 0)  # mean of x_i
	mj = K.expand_dims(K.mean(z, axis=0), 1)  # mean of z_j
	vj = K.expand_dims(K.var(z, axis=0) +  EPS, 1)  # sigma_j^2
	vi = K.expand_dims(K.var(x, axis=0) + EPS, 0)  # sigma_i^2
	V = div_n(K.dot(K.transpose(z-K.transpose(mj)), x- mi))
	rho = V / K.sqrt(vi*vj)
	# h(Xi) = .5 log (Xi2)
	return -.5*K.log(1-rho**2)#j i


def gaussian_cond_ent(x_, z, invert_sigmoid = True, binary_z = False): # returns j i matrix of I(X_i:Y_j)
	batch_size = K.cast(K.shape(x_)[0], x_.dtype)  # This is a node tensor, so we can't treat as integer
	div_n = Lambda(lambda v: v / batch_size)  
	size = K.cast(K.shape(x_)[1], dtype='int32')

	if invert_sigmoid:
		x = K.clip(x_, EPS, 1-EPS)
		log_det_jac = -K.mean(K.log(K.clip(tf.multiply(x, 1-x), EPS, 1-EPS)))
		x = K.log(x) - K.log(1-x)
		if binary_z:
			z = K.clip(z, EPS, 1-EPS)
			z = K.log(z) - K.log(1-z)
			# z is likely continuous, but invert sigmoid if binary		
	else:
		log_det_jac = 0

	mi = K.expand_dims(K.mean(x, axis=0), 0)  # mean of x_i
	mj = K.expand_dims(K.mean(z, axis=0), 1)  # mean of z_j
	vj = K.expand_dims(K.var(z, axis=0)+EPS, 1)  # sigma_j^2
	vi = K.expand_dims(K.var(x, axis=0)+EPS, 0)  # sigma_i^2
	V = div_n(K.dot(K.transpose(z-K.transpose(mj)), x- mi)) #jxi
	rho = V / K.sqrt(vi*vj)
	# h(Xi) = .5 log (Xi2)    
	# h(Xi | Yj) = .5 log (2pi e)* var2 (= xi2 - xiy*yxi / yj2)
	cond_var = vi - tf.divide(V**2, vj)
	return .5*(1.+np.log(2 * np.pi) + K.log(cond_var+EPS)) #- log_det_jac
	#return -.5*K.log(K.clip(1-rho**2, epsilon, 1-epsilon) )+ .5*(1.+np.log(2 * np.pi)+K.log(K.clip(vi, epsilon, 1-epsilon)))


''' ***** GIVES 0s IMMEDIATELY *****
		-numerical EPS?
'''
def error_entropy_mi(batch = 256, n = 784, m = 100, skew = True, kurt = True):
	#batch, n = merged_decode.get_shape()
	#n = int(n)
	#batch = int(batch)
	#div_n = Lambda(lambda v: v / batch) 
	def my_loss(x_true, merged_decode): 
		x_decode = merged_decode[:, :n]
		z = merged_decode[:, n:] 
		size = int(merged_decode.get_shape()[1]) - n

		error = x_true - x_decode
		error = error - K.mean(error, axis = 0)
		error_var = K.mean(error**2, axis = 0) # 1 x i
		error_ent = error_entropy(x_true, x_decode, skew = skew, kurt = kurt)
		z = z - K.mean(z, axis = 0)
		cov_z_1 = 1./batch*(K.dot(K.transpose(z), z))
		cov_error = 1./batch*(K.dot(K.transpose(error), error))
		cross_cov = 1./batch*(K.dot(K.transpose(z), error)) #j x i
		
		for i in range(n):
			cov_z_err_1 = tf.expand_dims(K.dot(cross_cov[:,i][:,np.newaxis], K.transpose(cross_cov[:,i][:,np.newaxis])), axis=0) #1 x m x m hopefully
			cov_z_err = cov_z_err if i == 0 else merge([cov_z_err, cov_z_err_1], mode ='concat', concat_axis = 0)
			#i x j x j to take mutual informations (H(Y|xi - g(Y) => cov matrx for each xi)
			cov_z = K.expand_dims(cov_z_1, 0) if i == 0 else merge([cov_z, K.expand_dims(cov_z_1, 0)], mode ='concat', concat_axis = 0) #will become i x m x m for broadcasting with i x 1 x 1 error_var 
		
		if size > 0:
			#empty tensor, fill with vector
			err_var = np.ones((n,m,m))
			#err_var_1 = K.expand_dims(K.expand_dims(error_var, -1), -1)
			for i in range(m):
				err_var[i,:,:].fill(error_var[i].eval())
			err_var = K.variable(err_var)
			#err_var = err_var_1 if i == 0 else merge([err_var, err_var_1], mode ='concat', concat_axis = 0)
			#error_var = K.expand_dims(K.expand_dims(error_var, -1), -1) # i x 1 x 1
			#err_var_1 = K.expand_dims(K.expand_dims(error_var, -1), -1)

			#print tf.matrix_determinant(K.variable(np.array([[2, 3],[4, 1]])))
			#print 'cov_z shape: ', cov_z.get_shape(), ' err_var shape: ', err_var.get_shape()

			mi = .5*K.log(tf.matrix_determinant(cov_z)+EPS) - .5*K.log(tf.matrix_determinant(cov_z - tf.div(cov_z_err, err_var))+EPS)
			
			return -K.sum(mi) #error_ent - K.sum(mi) 
		else: 
			return K.variable(np.array([0]))
	return my_loss

def error_entropy(x_true, x_decode, invert_sigmoid = False, subtract_log_det=False, skew = True, kurt = True):
	H_NU = 0.5 * (1. + np.log(2 * np.pi))  # Entropy of standard gaussian
	# IF INVERT_SIGMOID = TRUE, x_true binary, invert sigmoid to do gaussian entropy estimation in domain R  
	if invert_sigmoid:
		x_true = K.clip(x_true, EPS, 1-EPS)
		log_det_jac = -K.mean(K.log(K.clip(tf.multiply(x_true, 1-x_true), EPS, 1-EPS)))
		# DOES IT CANCEL WITH MAX(XI:ZJ)?
		x_true = K.log(x_true) - K.log(1-x_true)
		# appropriate?  NO
		#x_decode = K.clip(x_decode, EPS, 1-EPS)
		#x_decode = K.log(x_decode) - K.log(1-x_decode)
	else:
		log_det_jac = 0
	#log_det_jac = -K.mean(K.log(K.clip(tf.multiply(x_true, 1-x_true), EPS, 1-EPS))) if invert_sigmoid else 0 
	# derivative of log (p / 1-p ) = 1 / p(1-p), log (this) = - log p(1-p) )
	

	z = x_true - x_decode
	batch_mean = K.mean(z, axis=0)
	z = z - batch_mean
	batch_var = K.mean(z**2, axis=0)
	out = .5*K.sum(1+ K.log(2*np.pi*(batch_var+EPS))) 
	out = out - log_det_jac if subtract_log_det else out
	if skew or kurt:
	    z = tf.div(z, K.sqrt(batch_var + EPS))  # TODO: tune. Avoids numerical errors, but just adding EPS leads to infinities (why?)
	if skew:
	    k1 = 36. / (8. * np.sqrt(3.) - 9.)
	    B = K.mean(tf.multiply(z, tf.exp(- 0.5 * tf.square(z))))
	    out -= k1 * K.sum(tf.square(B))
	if kurt:
	    k2 = 1. / (2. - 6. / np.pi)
	    k3 = np.sqrt(2. / np.pi)
	    A = K.mean(tf.abs(z))
	    out -= k2 * K.sum(tf.square(A - k3))
	return out


def info_dropout_kl(n, batch = 256): #SOFTPLUS ONLY
    print 'LOSS FUNCTION: information dropout'
    def my_loss(x, merged_decode):
        #beta_ = beta_cb.get_beta()	
		b, n = merged_decode.get_shape()
		n = int(n/2)			
		z_output = merged_decode[:, n:]
		z_log_noise= merged_decode[:, :n]
    	
		#info dropout estimation of I(X:Z)
		mu_zj = K.mean(K.exp(z_output), axis=0)
		var_zj = K.var(K.exp(z_output), axis=0)
		alpha_2 = K.exp(z_log_noise) # variance of log_normal noise (mean 0).  alpha^2 in eq 5 of paper
		kl_loss = K.sum(.5/var_zj*(alpha_2 + mu_zj**2)-K.log(K.sqrt(alpha_2) / K.sqrt(var_zj))-.5)
		#should this be mean rather than sum?
		#kl_loss = K.sum(1.0/2*(K.exp(z_log_noise))-tf.log(tf.abs(K.exp(z_log_noise/2)))-.5, axis = 1) #prior = 0,1.. K.log(np.divide(z_log_noise, prior_std))       
		return 1.0/batch*kl_loss 
    return my_loss


def screening(n = 784, alpha=100, skew = True, kurt = True):
	print 'LOSS FUNCTION: Screening'
	def my_loss(x_true, merged_decode):
		#batch, n = merged_decode.get_shape()
		x_decode = merged_decode[:, :n]
		z = merged_decode[:, n:]

		#expected to be j x i?
		h_xi_zj = gaussian_cond_ent(x_true, z) # NOW this is a conditional entropy not MI
		#log_det_jac_xi_zj = 

		#mi_ji = gaussian_mi(x_true, z)

		#OLD calculation of h(Xi - g(Z)) term within screening loss
		#xi_entropy_est = error_entropy(x_true, 0, skew = skew, kurt = kurt)
		h_xi_z = error_entropy(x_true, x_decode, invert_sigmoid = True, subtract_log_det=False, skew = skew, kurt = kurt)
		# TAKING MIN
		min_xi_zj = tf.divide(K.sum(tf.multiply(h_xi_zj, K.exp(-alpha*h_xi_zj)), axis = 0), K.sum(K.exp(-alpha*h_xi_zj), axis = 0))

		#return -K.sum(h_xi_z) + K.sum(min_xi_zj)
		return -K.sum(h_xi_z) + K.sum(min_xi_zj)
		#Alternatively, in models.fit(), update_loss_weights to set recon loss weight to 1-beta
	return my_loss

def binary_ce(merged = True): #lagr = 1, 
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

	#ci = K.mean(tf.multiply(x_decode, K.log(x_decode)), axis = 0) - K.mean(tf.multiply(x_decode, K.log(ci_decode+EPS)), axis = 0)
	#ci = ci + K.mean(tf.multiply((1-x_decode), tf.log1p(-x_decode)), axis = 0) - K.mean(tf.multiply((1-x_decode),tf.log1p(-ci_decode-EPS)), axis = 0)
	#return K.sum(ci)
	#return K.mean(objectives.kullback_leibler_divergence(x_decode, ci_decode))
	ci = -objectives.binary_crossentropy(x_decode, x_decode) + objectives.binary_crossentropy(x_decode, ci_decode)
	#ci = ci + objectives.binary_crossentropy(x, ci_decode)
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
			full_var = K.var(x_true - x_decode, axis = 0) 
		else: 
			full_var = K.exp(log_noise)
		ci_var = 1/(1+ci_R)
		# ci : full -> ci = x_decode -> ci_decode
		ci = .5*((full_var/ci_var) + K.mean(ci_decode - x_decode, axis = 0)**2 / ci_var - 1 + K.log(ci_var/full_var))
		return K.mean(ci) #should be sum, but too big!s
	return my_loss
		# gaussian case... similarly for log-normal (INVOLVES MATRIX INVERSION)
		# cond_mu = K.transpose(K.dot(K.transpose(V), inv(Vj)))
	 	# full_mu = mi + K.dot(z-K.transpose(mj), cond_mu)
	 	# full_var = vi - diag_part(K.dot(K.transpose(cond_mu), V))
	 	# full_var = tf.multiply(K.ones_like(full_mu), full_var)
		#gaussian: ci = .5*((full_var / ci_var) + K.mean(ci_decode - x_decode, axis = 0)**2 / ci_var - 1 + K.log(ci_var/full_var))
 
def sigmoid_ci(x, merged_decode): # used by CI encoder regularization in lieu of Gaussian
	# assumes both merged_decode are continuous (e.g. nn activations vs. gaussian z_ci).  approximate CI using binary formula
	batch, n = merged_decode.get_shape()
	n = int(n/2)
	x_decode = merged_decode[:, :n]
	ci_decode = merged_decode[:, n:]

	x_decode = K.sigmoid(x_decode)
	ci_decode = K.sigmoid(ci_decode)
	ci = -objectives.binary_crossentropy(x_decode, x_decode) + objectives.binary_crossentropy(x_decode, ci_decode)
	return n*K.mean(ci) # = sum over xi (objectives returns mean across x_i for batch)



# def q_constraint(betas, R):
# 	def my_loss(x_true, x_decode):
# 		q = output of CI beta layer = p(yj) prod p(xi | yj)

#def p_x_constraint:




def ci_wms_dec(z_log_noise = None, batch = 256): # for use as regularization for CI decoder
	def my_loss(x, z):
		#batch, m = z.get_shape()
		#batch = K.cast(batch, x.dtype)
		#div_n = Lambda(lambda v: v / batch)
		
		z = z - K.mean(z, axis = 0, keepdims = True)
		x = x - K.mean(x, axis = 0, keepdims = True)
		x_2 = K.var(x, axis = 0, keepdims = True) #1xi
		z_2 = K.var(z, axis = 0, keepdims = True) #1xj
		#zj_xi = div_n(K.dot(K.transpose(z), x))	
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


# FULL LOSSES (DEPRECATED)
def recon_plus_reg(beta_reg, recon_reg = 1, reg = 'binary_ci', recon = 'binary_crossentropy'):
	#unpack here rather than in binary_ci
	def my_loss(x_true, merged_decode):
		if reg == 'binary_ci':
			batch, n = merged_decode.get_shape()
			n = int(n/2)
			x_decode = merged_decode[:, :n]
			ci_decode = merged_decode[:, n:]
			#n = merged_decode.get_shape()[1]
			# SHOULD BE SUM?
			reg_loss = K.sum(binary_ci(x_true, merged_decode)) #unpacks to KL(x_decode, ci_decode)
			#reg_loss = reg_loss+K.mean(binary_ci(ci_decode, x_true, merged = False))
		if recon == 'binary_crossentropy':
			recon_loss = objectives.binary_crossentropy(x_true, x_decode) 
			#binary_ce(x_true, merged_decode)
		return recon_reg*recon_loss + beta_reg*reg_loss
	return my_loss

def info_dropout(beta, n, m, recon_loss = 'binary_crossentropy'): 
    print 'LOSS FUNCTION: information dropout'
    def my_loss(x, merged_decode):
        #beta_ = beta_cb.get_beta()				
        z_output = merged_decode[:, n+m:]
        z_log_noise= merged_decode[:, n:n+m]
        x_decode = merged_decode[:, :n]
        if recon_loss == 'binary_crossentropy':
        	recon = objectives.binary_crossentropy(x, x_decode)
        elif recon_loss == 'error_entropy':
        	xi_entropy_est = error_entropy(x, 0)
        	entropies = error_entropy(x, x_decode, skew = True, kurt = True)
        	recon = -K.sum((1-beta)*(xi_entropy_est - entropies))
    	
    	#info dropout estimation of I(X:Z)
    	mu_zj = K.mean(K.exp(z_output), axis=0)
    	var_zj = K.var(K.exp(z_output), axis=0)
    	var_alpha = K.exp(z_log_noise) # variance of log_normal noise (mean 0).  alpha^2 in eq 5 of paper
        kl_loss = K.sum(.5/var_zj*(var_alpha + mu_zj)-K.log(K.sqrt(var_alpha) / K.sqrt(var_zj))-.5)
        #kl_loss = K.sum(1.0/2*(K.exp(z_log_noise))-tf.log(tf.abs(K.exp(z_log_noise/2)))-.5, axis = 1) #prior = 0,1.. K.log(np.divide(z_log_noise, prior_std))       
        return recon + beta*kl_loss 
    return my_loss

def loss_plus_ci(beta, n, strategy ='ci_binary', recon_loss = 'binary_crossentropy', EPS = 10**-5):
	print 'LOSS FUNCTION: loss plus CI'
	print 'PARAMS: ', beta
	def my_loss(x_true, merged_decode):
		# KL p(x_i | y) || p_ci(x_i | y)
		# .5 ( tr(cov_ci inv * cov_p ) + (mu_ci - mu_p)T cov_ci inv (mu_ci - mu_p) - k + ln [(det cov_ci) / det (cov_p)])
		if strategy == 'cd_gauss':
			x_decode = merged_decode[:, :n]
			full_var = merged_decode[:, n:2*n]
			ci_decode = merged_decode[:, -2*n:-n]
			ci_var = merged_decode[:, -n:]

			# x_decode = merged_decode[:, :n]
			# full_var = merged_decode[:, n:n+1]
			# ci_decode = merged_decode[:, n+2:-1]
			# ci_var = merged_decode[:, -1]
			
			#cond_ent = error_entropy(x_decode, 0, skew = True, kurt = True)
			#ci_xent = tf.divide(x_decode - ci_decode, K.sqrt(ci_var)) # z score of p(xi|y) full gaussian under ci decoder
			#dist = tf.contrib.distributions.Normal(0.0, 1.0)
			#ci_xent = K.mean(K.log(dist.pdf(ci_xent)), axis =0)
			#ci = - ci_xent #cond_ent - ci_xent
			#full_var = joint gaussian p(xi, y), gaussian KL divergence formula
			
		elif strategy == 'ci_binary':
			x_decode = merged_decode[:, :n]
			ci_decode = merged_decode[:, n:]
			
			#p, p_ci
			ci = - objectives.binary_crossentropy(x_decode, x_decode) + objectives.binary_crossentropy(x_decode, ci_decode)
			#ci = - K.binary_crossentropy(x_decode, x_decode) + K.binary_crossentropy(x_decode, ci_decode)
			# sum(xi, y) p(xi|y) log p(xi|y) / p_ci(xi|y)
			# let x_i = 1 / 0 , y = all data points
			
			#ci = K.mean(tf.multiply(x_decode, K.log(x_decode)), axis = 0) - K.mean(tf.multiply(x_decode, K.log(ci_decode+EPS)), axis = 0)
			#ci = ci + K.mean(tf.multiply((1-x_decode), tf.log1p(-x_decode)), axis = 0) - K.mean(tf.multiply((1-x_decode),tf.log1p(-ci_decode+EPS)), axis = 0)
			
		#full KL divergence for Gaussian case? 
		if recon_loss == 'binary_crossentropy':
			#recon = K.mean(tf.multiply(x_true, K.log(x_true)), axis = 0) - K.mean(tf.multiply(x_true, K.log(x_decode)), axis = 0)
			#recon = recon + K.mean(tf.multiply((1-x_true), tf.log1p(-x_true)), axis = 0) - K.mean(tf.multiply((1-x_true),tf.log1p(-x_decode)), axis = 0)
			reconstruction = objectives.binary_crossentropy(x_true, x_decode)
			#reconstruction = objectives.binary_crossentropy(tf.nn.softmax(x_true, dim = 1), tf.nn.softmax(x_decode, dim=1))
		#elif:
		return beta*K.sum(ci)+reconstruction  
	return my_loss

def xi_ent_plus_gk(beta, alpha, n, strategy = 'ms_gauss'): #alpha, m, n, strategy = 'ms_gauss'):
	print 'LOSS FUNCTION: xi_ent plus GK'
	def my_loss(x_true, merged_decode):
		x_decode = merged_decode[:, :n]
		z = merged_decode[:, n:]
		mi_ji = gaussian_mi(x_true, z)
		xi_entropy_est = error_entropy(x_true, 0, skew = False, kurt = False)
		entropies = error_entropy(x_true, x_decode, skew = False, kurt = False)
		# smooth_max with param alpha
		max_ji = tf.divide(K.sum(tf.multiply(mi_ji, K.exp(alpha*mi_ji)), axis = 0), K.sum(K.exp(alpha*mi_ji), axis = 0))
		# beta_i
		return (1.0-beta)*(entropies)-beta*K.sum(max_ji)
	return my_loss




def ci_plus_wms(x, y, x_2, y_2, yj_xi):
	print 'LOSS FUNCTION: CI plus WMS'
	#for ci distributions... note p_ci(y) = 1/(1+R)
	#wms = -tc(y:xi) = E(p_ci(xi, y)) log p_ci(y)) / prod (p_yj)
	#cov_j_i  not equal A / R

	cov_j_i = -K.divide(yj_xi**2, x_2) + y_2
	A_i = tf.multiply(x_2**-1, 1+ K.sum(tf.divide(yj_xi**2, K.dot(y_2, x_2) - yj_xi**2),  axis =0)) #j xi
	mu_ci_ji = tf.multiply(tf.divide(1, A_i),  tf.divide(yj_xi, K.dot(y_2, x_2) - yj_xi**2))
	prod_cov_j_i = tf.reduce_prod(cov_j_i, axis = 0) 	
	exponent = tf.multiply(A_i, (x - K.dot(mu_ci_ji, y))**2) - K.divide(K.dot(mu_ci_ji, y)**2, A_i) + K.sum(y_2, axis =1)
	joint = (2*np.pi)**(-.5)*tf.multiply(prod_cov_j_i ** -1, x_2 ** -1)*K.exp(-.5*exponent)
	dist = tf.contrib.distributions.Normal(0.0, 1.0)
	prod_yj = tf.reduce_prod(dist.pdf(tf.divide(y, y_2)))
	#expectation over batch =  p_ci(xi,y)
	#K.sum(tf.multiply(joint, K.log(prod_yj)-K.log(A_i)



def anneal_model(model, betas, sched, n_epoch, loss_function, batch_size, x_train, x_test, strategy, m = 0, optimizer = 'Adam', alpha = 10):
	k = 0
	ep = 0
	while ep < n_epoch:
	    if ep == sched[k]:
	        beta = betas[k]
	        if k < len(sched)-1:
	            k = k+1
	            ep = sched[k]
	            epochs = sched[k]-sched[k-1]
	        else:
	            ep = n_epoch
	            epochs = n_epoch - sched[k]
		if strategy == 'screening':
			print 'Params: ', beta, alpha
			model.compile(optimizer=optimizer, loss = loss_function(beta, alpha, x_train.shape[1], strategy))
		elif strategy.startswith('info_dropout'):
			if strategy == 'info_dropout_tc':
				model.compile(optimizer=optimizer, loss = loss_function(beta, x_train.shape[1], m, recon_loss = 'error_entropy'))
			else:	
				model.compile(optimizer=optimizer, loss = loss_function(beta, x_train.shape[1], m))
		elif strategy == 'ci_binary':
				model.compile(optimizer=optimizer, loss = loss_function(beta))
		else: #xi_ent_plus_gk or loss_plus_ci
			model.compile(optimizer=optimizer, loss = loss_function(beta, x_train.shape[1], strategy))
	    model.fit(x_train, x_train,
	                 shuffle=True,
	                 nb_epoch=epochs,
	                 batch_size=batch_size,
	                 validation_data=(x_test, x_test))
	return model

