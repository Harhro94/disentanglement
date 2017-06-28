"""Custom layers specified in keras style."""

from keras import backend as K
from keras.layers import Lambda, merge, Dense
from keras import activations
from keras.engine.topology import Layer
import numpy as np
from tensorflow import matrix_inverse, diag_part
import tensorflow as tf

#for use in Lambda layers
def sample_vae(args):
    z_mean, z_log_var = args
    batch, features = z_mean.get_shape()
    if batch > 0:
      epsilon = K.random_normal(shape=(batch, features), mean=0.,
                              std=epsilon_std)
      return z_mean + K.exp(z_log_var / 2) * epsilon
    else: 
      return z_mean
    

def sample_info_dropout(args):
    z_mean, z_log_noise = args
    #sample single from each z_noise entry
    batch, features = z_mean.get_shape()
    #batch = K.cast(K.shape(z_mean)[0], 'int32') 
    #features = K.cast(K.shape(z_mean)[1], 'int32') 
    #print 'z mean shape:' , z_mean.get_shape()
    if batch > 0:
      epsilon = K.random_normal(shape=(batch, features), mean=0.,
                              std=1.0)
    else:
      epsilon = 0
      return z_mean
    # f(x) * exp( mean = 0 + std normal sample * exp( log sqrt(Var) ))
    # return K.in_train_phase(tf.multiply(z_mean, K.exp(tf.multiply(K.sqrt(K.exp(z_log_noise)), epsilon))), z_mean)
    return tf.multiply(z_mean, K.log(tf.multiply(K.sqrt(K.exp(z_log_noise)), epsilon)))
    # K.in_train_phase(tf.multiply(z_mean, K.log(tf.multiply(K.exp(z_log_noise/2), epsilon))), z_mean)

def identity_layer(x):
    return x

def dense_k(x, layers, activation = 'softplus', initializer = 'glorot_uniform', decoder = False, leave_last = False):
    # Multi-layer default encoder
    z_j = x
    for j in range(len(layers)):
        if decoder:
          z_j = Dense(layers[j], activation = activation, init= initializer, name='decoder{}'.format(j+1))(z_j)
        else:
          if j == len(layers)-1 and not leave_last:
            z_j = Dense(layers[j], activation = activation, init= initializer, name='z_activation')(z_j)
          else:
            z_j = Dense(layers[j], activation = activation, init= initializer, name='encoder{}'.format(j+1))(z_j)

    return z_j
    
def dense_intermediate(x, layers, input_layer, encoder_add, activation = 'softplus'):
    z_j = input_layer
    for j in range(len(layers)):
        z_j = Dense(layers[j], activation=activation, name='encoder{}'.format(j+1+encoder_add))(z_j)
    return z_j

def sigmoid_weights(weights): 
    #duplicates work done in SigmoidFull layer.  hack for transferring pretrained ci weights to full nn
    def get_mean_sig(mj, vj, pi, V, S, epsilon=.001):
        mu1 = V / pi  # mu_xi=1^j
        mu0 = (mj - V) / (1 - pi)  # mu_xi=0^j
        sig1 = S / pi - K.square(V / pi) + epsilon  # sig2_xi=1^j
        sig0 = (vj - S) / (1. - pi) - K.square((mj - V) / (1. - pi)) + epsilon  # sig2_xi=0^j
        return mu0, mu1, sig0, sig1
    mu0, mu1, sig0, sig1 = get_mean_sig(weights[3], weights[4], weights[2], weights[0], weights[1])
    weights_ji = np.multiply(weights[4]**.5, (np.divide(mu1, sig1) - np.divide(mu0 , sig0)))     
    mean_i =  np.squeeze(weights[2].T)
    return weights_ji, mean_i


class DecodeSigmoidSimple(Layer):
    """ This decoder assumes a continuous input and a binary output. It fixes the form of the
    transformation based on an assumption of conditional independence.
    It requires access to the true decoded layer at training time. In keras, you need to at it to the input of
    layer like this:
    merged_vector = merge([z, x], mode='concat', concat_axis=-1)
    z = ms.DecodeSimple(original_dim, name='decoder1')(merged_vector)
    """
    def __init__(self, output_dim, epsilon=0.001, momentum=0.99, **kwargs):
        self.output_dim = output_dim
        self.epsilon = epsilon
        self.momentum = momentum
        self.uses_learning_phase = True
        super(DecodeSigmoidSimple, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.size = input_shape[1] - self.output_dim
        # These are running mean variables, stored for use in test phase.
        self.Vr = self.add_weight(shape=(self.size, self.output_dim),
                                 initializer='zero',
                                 trainable=False,
                                  name='{}_Vr'.format(self.name))
        self.pir = self.add_weight(shape=(1, self.output_dim),
                                 initializer='zero',
                                 trainable=False,
                                  name='{}_pir'.format(self.name))
        self.mjr = self.add_weight(shape=(self.size, 1),
                                 initializer='zero',
                                 trainable=False,
                                  name='{}_mjr'.format(self.name))
        self.vjr = self.add_weight(shape=(self.size, 1),
                                 initializer='one',
                                 trainable=False,
                                  name='{}_vjr'.format(self.name))
        super(DecodeSigmoidSimple, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x_cat, mask=None):
        # For some reason, we have to concatenate vectors to feed them using "merge" in keras
        z = x_cat[:, :self.size]
        x = x_cat[:, self.size:]
        batch_size = K.cast(K.shape(x)[0], x.dtype)  # This is a node tensor, so we can't treat as integer
        div_n = Lambda(lambda v: v / batch_size)  # Dividing by batch size is an operation on unknown tensor

        # batch statistics
        pi = K.expand_dims(K.clip(K.mean(x, axis=0), self.epsilon, 1. - self.epsilon), 0)  # p(xi = 1)
        mj = K.expand_dims(K.mean(z, axis=0), 1)  # mean of z_j
        vj = K.expand_dims(K.var(z, axis=0) + self.epsilon, 1)  # sigma_j^2
        V = div_n(K.dot(K.transpose(z), x))  # j i

        self.add_update([K.moving_average_update(self.Vr, V, self.momentum),
                         K.moving_average_update(self.pir, pi, self.momentum),
                         K.moving_average_update(self.mjr, mj, self.momentum),
                         K.moving_average_update(self.vjr, vj, self.momentum)
                         ], x_cat)
        V = K.in_train_phase(V, self.Vr)
        pi = K.in_train_phase(pi, self.pir)
        mj = K.in_train_phase(mj, self.mjr)
        vj = K.in_train_phase(vj, self.vjr)

        mu_diff = (V - mj * pi) / (pi * (1 - pi))  # difference between mu_xi=1^j - mu_xi=0^j
        mu_mean = 0.5 * (V / pi + (mj - V) / (1 - pi))  # average of means
        out = K.log(pi) - K.log(1. - pi) + K.dot(z, mu_diff / vj) - K.sum(mu_diff * mu_mean / vj, 0, keepdims=True)
        return K.sigmoid(out)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_w(self):
        # For visualization
        mu_diff = (self.Vr - self.mjr * self.pir) / (self.pir * (1 - self.pir))  # difference between mu_xi=1^j - mu_xi=0^j
        return mu_diff / K.sqrt(self.vjr)

class DecodeSigmoidFull(Layer):
    """ This decoder assumes a continuous input and a binary output. It fixes the form of the
    transformation based on an assumption of conditional independence.
    It requires access to the true decoded layer at training time. In keras, you need to at it to the input of
    layer like this:
    merged_vector = merge([z, x], mode='concat', concat_axis=-1)
    z = ms.DecodeSimple(original_dim, name='decoder1')(merged_vector)
    """
    def __init__(self, output_dim, epsilon=0.001, momentum=0.99, **kwargs):
        self.output_dim = output_dim
        self.epsilon = epsilon
        self.momentum = momentum
        self.uses_learning_phase = True
        super(DecodeSigmoidFull, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.size = input_shape[1] - self.output_dim
        # These are running mean variables, stored for use in test phase.
        self.Vr = self.add_weight(shape=(self.size, self.output_dim),
                                 initializer='zero',
                                 trainable=False,
                                  name='{}_Vr'.format(self.name))
        self.Sr = self.add_weight(shape=(self.size, self.output_dim),
                                  initializer='one',
                                  trainable=False,
                                  name='{}_Sr'.format(self.name))
        self.pir = self.add_weight(shape=(1, self.output_dim),
                                 initializer='zero',
                                 trainable=False,
                                  name='{}_pir'.format(self.name))
        self.mjr = self.add_weight(shape=(self.size, 1),
                                 initializer='zero',
                                 trainable=False,
                                  name='{}_mjr'.format(self.name))
        self.vjr = self.add_weight(shape=(self.size, 1),
                                 initializer='one',
                                 trainable=False,
                                  name='{}_vjr'.format(self.name))
        super(DecodeSigmoidFull, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x_cat, mask=None):
        # For some reason, we have to concatenate vectors to feed them using "merge" in keras
        z = x_cat[:, :self.size]
        x = K.clip(x_cat[:, self.size:], self.epsilon, 1. - self.epsilon)
        batch_size = K.cast(K.shape(x)[0], x.dtype)  # This is a node tensor, so we can't treat as integer
        div_n = Lambda(lambda v: v / batch_size)  # Dividing by batch size is an operation on unknown tensor

        # batch statistics
        pi = K.expand_dims(K.mean(x, axis=0), 0)  # p(xi = 1)
        mj = K.expand_dims(K.mean(z, axis=0), 1)  # mean of z_j
        vj = K.expand_dims(K.mean(K.square(z), axis=0), 1)  # expectation of z^2
        V = div_n(K.dot(K.transpose(z), x))  # j i
        S = div_n(K.dot(K.transpose(K.square(z)), x))  # j i

        self.add_update([K.moving_average_update(self.Vr, V, self.momentum),
                         K.moving_average_update(self.Sr, S, self.momentum),
                         K.moving_average_update(self.pir, pi, self.momentum),
                         K.moving_average_update(self.mjr, mj, self.momentum),
                         K.moving_average_update(self.vjr, vj, self.momentum)
                         ], x_cat)
        V = K.in_train_phase(V, self.Vr)
        S = K.in_train_phase(S, self.Sr)
        pi = K.in_train_phase(pi, self.pir)
        mj = K.in_train_phase(mj, self.mjr)
        vj = K.in_train_phase(vj, self.vjr)

        mu0, mu1, sig0, sig1 = self.get_mean_sig(mj, vj, pi, V, S)

        out = (K.log(pi) - K.log(1. - pi)
               - 0.5 * K.sum(K.log(sig1) - K.log(sig0), 0)
               + 0.5 * K.sum(K.square(mu0) / sig0 - K.square(mu1) / sig1, 0)
               + K.dot(z, mu1 / sig1 - mu0 / sig0)
               + 0.5 * K.dot(K.square(z), 1. / sig0 - 1. / sig1)
              )
        return K.sigmoid(out)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_mean_sig(self, mj, vj, pi, V, S):
        mu1 = V / pi  # mu_xi=1^j
        mu0 = (mj - V) / (1 - pi)  # mu_xi=0^j
        sig1 = S / pi - K.square(V / pi) + self.epsilon  # sig2_xi=1^j
        sig0 = (vj - S) / (1. - pi) - K.square((mj - V) / (1. - pi)) + self.epsilon  # sig2_xi=0^j
        return mu0, mu1, sig0, sig1

    def get_mean_j(self):
        return self.mjr

    def get_w(self):
        # For visualization
        # WHY QUALITATIVELY SO DIFFERENT THAN GAUSSIAN? on/off, but not as intense
        mu0, mu1, sig0, sig1 = self.get_mean_sig(self.mjr, self.vjr, self.pir, self.Vr, self.Sr)
        return K.sqrt(self.vjr) * (mu1 / sig1 - mu0 / sig0)

class DecodeB2B(Layer):
    """ This decoder assumes a binary input (values in [0,1] interpreted as probabilities)
     and a binary output (values in [0,1]). It fixes the form of the
    transformation based on an assumption of conditional independence.
    It requires access to the true decoded layer at training time. In keras, you need to at it to the input of
    layer like this:
    merged_vector = merge([z, x], mode='concat', concat_axis=-1)
    z = ms.DecodeSimple(original_dim, name='decoder1')(merged_vector)
    """
    def __init__(self, output_dim, epsilon=0.001, momentum=0.99, **kwargs):
        self.output_dim = output_dim
        self.epsilon = epsilon
        self.momentum = momentum
        self.uses_learning_phase = True
        super(DecodeB2B, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.size = input_shape[1] - self.output_dim
        # These are running mean variables, stored for use in test phase.
        self.Vr = self.add_weight(shape=(self.size, self.output_dim),
                                 initializer='zero',
                                 trainable=False,
                                  name='{}_Vr'.format(self.name))
        half_init = lambda shape, name: K.variable(0.5 * np.ones(shape), name=name)
        self.pxr = self.add_weight(shape=(1, self.output_dim),
                                 initializer=half_init,
                                 trainable=False,
                                  name='{}_pxr'.format(self.name))
        self.pyr = self.add_weight(shape=(self.size, 1),
                                 initializer=half_init,
                                 trainable=False,
                                  name='{}_pyr'.format(self.name))
        super(DecodeB2B, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x_cat, mask=None):
        # For some reason, we have to concatenate vectors to feed them using "merge" in keras
        x_cat = self.epsilon + (1 - 2. * self.epsilon) * x_cat  #K.clip(x_cat, self.epsilon, 1 - self.epsilon)  # Avoid NANs
        z = x_cat[:, :self.size]
        x = x_cat[:, self.size:]
        batch_size = K.cast(K.shape(x)[0], x.dtype)  # This is a node tensor, so we can't treat as integer
        div_n = Lambda(lambda v: v / batch_size)  # Dividing by batch size is an operation on unknown tensor

        # batch statistics
        px = K.expand_dims(K.mean(x, axis=0), 0)  # p(xi = 1)
        py = K.expand_dims(K.mean(z, axis=0), 1)  # mean of z_j
        V = div_n(K.dot(K.transpose(z), x))  # j i

        self.add_update([K.moving_average_update(self.Vr, V, self.momentum),
                         K.moving_average_update(self.pxr, px, self.momentum),
                         K.moving_average_update(self.pyr, py, self.momentum)
                         ], x_cat)
        V = K.in_train_phase(V, self.Vr)
        px = K.in_train_phase(px, self.pxr)
        py = K.in_train_phase(py, self.pyr)
        eta1 = V / px
        eta0 = (py - V) / (1 - px)
        W = K.log(eta1) - K.log(1 - eta1) + K.log(1 - eta0) - K.log(eta0)
        out = K.log(px) - K.log(1. - px) + K.dot(z, W) + K.sum(K.log(1. - eta1) - K.log(1. - eta0), 0, keepdims=True)
        return K.sigmoid(out)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_w(self):
        px, py, V = self.pxr, self.pyr, self.Vr
        eta1 = V / px
        eta0 = (py - V) / (1 - px)
        return K.log(eta1) - K.log(1 - eta1) + K.log(1 - eta0) - K.log(eta0)

class Generic(Layer):
    """ A generic dense, autoencoder layer with a small twist. We regularize the W's according to:
    Eq. ? in arxiv/...
    """
    def __init__(self, output_dim, activation=None, **kwargs):
        self.output_dim = output_dim
        self.activation = activations.get(activation)
        self.uses_learning_phase = False
        super(Generic, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.size = input_shape[1]
        self.W = self.add_weight(shape=(self.size, self.output_dim),
                                 initializer='normal',
                                 trainable=True,
                                 name='{}_W'.format(self.name))
        self.beta = self.add_weight(shape=(self.output_dim,),
                                    initializer='zero',
                                    trainable=True,
                                    name='{}_beta'.format(self.name))
        self.gamma = self.add_weight(shape=(self.output_dim,),
                                    initializer='one',
                                    trainable=False,
                                    name='{}_gamma'.format(self.name))
        super(Generic, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, z, mask=None):
        W = self.W / (1. + 0.5 *
                      K.sum(K.sqrt(1. + 4. * K.square(self.W)) - 1., 0, keepdims=True))
        out = K.reshape(self.gamma, [1, self.output_dim]) * K.dot(z, W) + K.reshape(self.beta, [1, self.output_dim])
        return self.activation(out)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)

class TiedEncodeDecode(Layer):
    """ A regular dense layer that encodes and decodes using tied weights.
    """
    def __init__(self, output_dim, latent_dim, activation=None, **kwargs):
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.activation = activations.get(activation)
        self.uses_learning_phase = False
        super(TiedEncodeDecode, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(shape=(self.output_dim, self.latent_dim),
                                 initializer='normal',
                                 trainable=True,
                                 name='{}_W'.format(self.name))
        self.be = self.add_weight(shape=(self.latent_dim,),
                                    initializer='zero',
                                    trainable=True,
                                    name='{}_b_encode'.format(self.name))
        self.bd = self.add_weight(shape=(self.output_dim,),
                            initializer='zero',
                            trainable=True,
                            name='{}_b_decode'.format(self.name))
        super(TiedEncodeDecode, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, z, mask=None):
        out = K.dot(z, self.W) + K.reshape(self.be, [1, self.latent_dim])
        out = self.activation(out)
        out = K.dot(out, K.transpose(self.W)) + K.reshape(self.bd, [1, self.output_dim])
        return self.activation(out)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_w(self):
        return self.W

class DecodeGaussian(Layer):
    """ This decoder assumes a continuous input and a continuous output. It fixes the form of the
    transformation based on an assumption of conditional independence.
    It requires access to the true decoded layer at training time. In keras, you need to at it to the input of
    layer like this:
    merged_vector = merge([z, x], mode='concat', concat_axis=-1)
    z = ms.DecodeGaussian(original_dim, name='decoder1')(merged_vector)
    """
    def __init__(self, output_dim, return_r = False, epsilon=0.001, momentum=0.99, **kwargs):
        self.output_dim = output_dim
        self.epsilon = epsilon
        self.momentum = momentum
        self.uses_learning_phase = True
        self.return_r = return_r
        super(DecodeGaussian, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.size = input_shape[1] - self.output_dim
        # These are running mean variables, stored for use in test phase.
        self.Vr = self.add_weight(shape=(self.size, self.output_dim),
                                  initializer='zero',
                                  trainable=False,
                                  name='{}_Vr'.format(self.name))
        self.mir = self.add_weight(shape=(1, self.output_dim),
                                   initializer='zero',
                                   trainable=False,
                                   name='{}_mir'.format(self.name))
        self.mjr = self.add_weight(shape=(self.size, 1),
                                   initializer='zero',
                                   trainable=False,
                                   name='{}_mjr'.format(self.name))
        self.vir = self.add_weight(shape=(1, self.output_dim),
                                   initializer='one',
                                   trainable=False,
                                   name='{}_vir'.format(self.name))
        self.vjr = self.add_weight(shape=(self.size, 1),
                                   initializer='one',
                                   trainable=False,
                                   name='{}_vjr'.format(self.name))
        self.R = K.placeholder(shape = (self.size, 1))
        super(DecodeGaussian, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x_cat, mask=None):
        # For some reason, we have to concatenate vectors to feed them using "merge" in keras
        z = x_cat[:, :self.size]
        x = x_cat[:, self.size:]
        batch_size = K.cast(K.shape(x)[0], x.dtype)  # This is a node tensor, so we can't treat as integer
        div_n = Lambda(lambda v: v / batch_size)  # Dividing by batch size is an operation on unknown tensor

        # batch statistics
        self.mi = K.expand_dims(K.mean(x, axis=0), 0)  # mean of x_i
        self.mj = K.expand_dims(K.mean(z, axis=0), 1)  # mean of z_j
        self.vj = K.expand_dims(K.var(z, axis=0) + self.epsilon, 1)  # sigma_j^2
        self.vi = K.expand_dims(K.var(x, axis=0) + self.epsilon, 0)  # sigma_i^2
        
        #CHANGE BACK
        #self.V = div_n(K.dot(K.transpose(z), x))  
        self.V = div_n(K.dot(K.transpose(z-K.transpose(self.mj)), x-self.mi))  # j i

        self.add_update([K.moving_average_update(self.Vr, self.V, self.momentum),
                         K.moving_average_update(self.mir, self.mi, self.momentum),
                         K.moving_average_update(self.mjr, self.mj, self.momentum),
                         K.moving_average_update(self.vjr, self.vj, self.momentum),
                         K.moving_average_update(self.vir, self.vi, self.momentum)
                         ], x_cat)
        V = K.in_train_phase(self.V, self.Vr)
        mi = K.in_train_phase(self.mi, self.mir)
        mj = K.in_train_phase(self.mj, self.mjr)
        vj = K.in_train_phase(self.vj, self.vjr)
        vi = K.in_train_phase(self.vi, self.vir)
        
        #CHANGE BACK
        #rho = (V - mi * mj) / K.sqrt(vi * vj)
        rho = V / K.sqrt(vi*vj)
        Q = rho / (1 - K.square(rho))
        self.R = K.sum(rho * Q, axis=0, keepdims=True)
        Q = Q / (1 + self.R)
        if self.return_r:
          return self.R
        else:
          return mi + K.sqrt(vi) * K.dot(K.transpose((K.transpose(z) - mj) / K.sqrt(vj)), Q)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_w(self):
        # For visualization
        rho = self.Vr / K.sqrt(self.vir * self.vjr)
        #rho = (self.Vr - self.mir * self.mjr) / K.sqrt(self.vir * self.vjr)
        Q = rho / (1 - K.square(rho))
        R = K.sum(rho * Q, axis=0, keepdims=True)
        Q = Q / (1 + R)
        return Q

    def get_R(self):
        return self.R

    def get_yj_means(self):
        return K.transpose(K.in_train_phase(self.mj, self.mjr))

    def get_yj_vars(self):
        return K.transpose(K.in_train_phase(self.vj, self.vjr))

    def get_prev_layer(self):
        return Layer
    #x_i = Wij * y_j = Wi * y
    #cov matrix for  h(x_i - g(y)) for minsyn decoder
    #gaussian y, true y, min entropy which pushes down h(x_i |y)

class GaussianMI(Layer):
    def __init__(self, output_dim, epsilon=0.001, **kwargs):
        self.output_dim = output_dim
        self.epsilon = epsilon
        self.uses_learning_phase = True
        super(GaussianMI, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.size = input_shape[1] - self.output_dim
        # These are running mean variables, stored for use in test phase.
        self.Vr = self.add_weight(shape=(self.size, self.output_dim),
                                  initializer='zero',
                                  trainable=False,
                                  name='{}_Vr'.format(self.name))
        self.mir = self.add_weight(shape=(1, self.output_dim),
                                   initializer='zero',
                                   trainable=False,
                                   name='{}_mir'.format(self.name))
        self.mjr = self.add_weight(shape=(self.size, 1),
                                   initializer='zero',
                                   trainable=False,
                                   name='{}_mjr'.format(self.name))
        self.vir = self.add_weight(shape=(1, self.output_dim),
                                   initializer='one',
                                   trainable=False,
                                   name='{}_vir'.format(self.name))
        self.vjr = self.add_weight(shape=(self.size, 1),
                                   initializer='one',
                                   trainable=False,
                                   name='{}_vjr'.format(self.name))
        self.R = K.placeholder(shape = (self.size, 1))
        super(GaussianMI, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x_cat, mask=None):
        # For some reason, we have to concatenate vectors to feed them using "merge" in keras
        z = x_cat[:, :self.size]
        x = x_cat[:, self.size:]
        batch_size = K.cast(K.shape(x)[0], x.dtype)  # This is a node tensor, so we can't treat as integer
        div_n = Lambda(lambda v: v / batch_size)  # Dividing by batch size is an operation on unknown tensor

        # batch statistics
        self.mi = K.expand_dims(K.mean(x, axis=0), 0)  # mean of x_i
        self.mj = K.expand_dims(K.mean(z, axis=0), 1)  # mean of z_j
        self.vj = K.expand_dims(K.var(z, axis=0) + self.epsilon, 1)  # sigma_j^2
        self.vi = K.expand_dims(K.var(x, axis=0) + self.epsilon, 0)  # sigma_i^2

        self.V = div_n(K.dot(K.transpose(z-K.transpose(self.mj)), x-self.mi))  # j i

        #self.add_update([K.moving_average_update(self.Vr, self.V, self.momentum),
        #                 K.moving_average_update(self.mir, self.mi, self.momentum),
        #                 K.moving_average_update(self.mjr, self.mj, self.momentum),
        #                 K.moving_average_update(self.vjr, self.vj, self.momentum),
        #                 K.moving_average_update(self.vir, self.vi, self.momentum)
        #                 ], x_cat)
        #V = K.in_train_phase(self.V, self.Vr)
        #mi = K.in_train_phase(self.mi, self.mir)
        #mj = K.in_train_phase(self.mj, self.mjr)
        #vj = K.in_train_phase(self.vj, self.vjr)
        #vi = K.in_train_phase(self.vi, self.vir)
        
        rho = self.V / K.sqrt(self.vi*self.vj)
        return [-.5*K.log(1-rho[j,:]**2) for j in range(self.size)] #jxi

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)

   # def get_w(self):
        # For visualization
   #     rho = self.V / K.sqrt(self.vi * self.vj)
   #     return -.5*K.log(1-rho**2)

class DecodeGaussianBeta(Layer):
    def __init__(self, output_dim, return_r = False, epsilon=0.001, momentum=0.99, **kwargs):
        self.output_dim = output_dim
        self.epsilon = epsilon
        self.momentum = momentum
        self.uses_learning_phase = True
        self.return_r = return_r
        super(DecodeGaussianBeta, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.size = input_shape[1] - self.output_dim
        # These are running mean variables, stored for use in test phase.
        self.Vr = self.add_weight(shape=(self.size, self.output_dim),
                                  initializer='zero',
                                  trainable=False,
                                  name='{}_Vr'.format(self.name))
        self.mir = self.add_weight(shape=(1, self.output_dim),
                                   initializer='zero',
                                   trainable=False,
                                   name='{}_mir'.format(self.name))
        self.mjr = self.add_weight(shape=(self.size, 1),
                                   initializer='zero',
                                   trainable=False,
                                   name='{}_mjr'.format(self.name))
        self.vir = self.add_weight(shape=(1, self.output_dim),
                                   initializer='one',
                                   trainable=False,
                                   name='{}_vir'.format(self.name))
        self.vjr = self.add_weight(shape=(self.size, 1),
                                   initializer='one',
                                   trainable=False,
                                   name='{}_vjr'.format(self.name))
        #or just i?
        self.betaji = self.add_weight(shape=(self.size, 1), 
                                   initializer='one',
                                   trainable=True,
                                   name='{}_betaij'.format(self.name)) #self.size
        self.R = K.placeholder(shape = (self.size, 1))
        super(DecodeGaussianBeta, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x_cat, mask=None):
        # For some reason, we have to concatenate vectors to feed them using "merge" in keras
        z = x_cat[:, :self.size]
        x = x_cat[:, self.size:]
        batch_size = K.cast(K.shape(x)[0], x.dtype)  # This is a node tensor, so we can't treat as integer
        div_n = Lambda(lambda v: v / batch_size)  # Dividing by batch size is an operation on unknown tensor

        # batch statistics
        mi = K.expand_dims(K.mean(x, axis=0), 0)  # mean of x_i
        mj = K.expand_dims(K.mean(z, axis=0), 1)  # mean of z_j
        vj = K.expand_dims(K.var(z, axis=0) + self.epsilon, 1)  # sigma_j^2
        vi = K.expand_dims(K.var(x, axis=0) + self.epsilon, 0)  # sigma_i^2
        V = div_n(K.dot(K.transpose(z), x))  # j i

        self.add_update([K.moving_average_update(self.Vr, V, self.momentum),
                         K.moving_average_update(self.mir, mi, self.momentum),
                         K.moving_average_update(self.mjr, mj, self.momentum),
                         K.moving_average_update(self.vjr, vj, self.momentum),
                         K.moving_average_update(self.vir, vi, self.momentum)
                         ], x_cat)
        V = K.in_train_phase(V, self.Vr)
        mi = K.in_train_phase(mi, self.mir)
        mj = K.in_train_phase(mj, self.mjr)
        vj = K.in_train_phase(vj, self.vjr)
        vi = K.in_train_phase(vi, self.vir)
        rho = (V - mi * mj) / K.sqrt(vi * vj)
        Q = tf.multiply(rho / (1 - K.square(rho)), self.betaji)
        self.R = K.sum(rho * Q, axis=0, keepdims=True)
        Q = Q / (1 + self.R)
        if self.return_r:
          return self.R
        else:
          return mi + K.sqrt(vi) * K.dot(K.transpose((K.transpose(z) - mj) / K.sqrt(vj)), Q)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_w(self):
        # For visualization
        rho = (self.Vr - self.mir * self.mjr) / K.sqrt(self.vir * self.vjr)
        Q = tf.multiply(self.betaji, rho) / (1 - K.square(rho))
        R = K.sum(rho * Q, axis=0, keepdims=True)
        Q = Q / (1 + R)
        return Q #, self.betaij

    def get_betas(self):
        return self.betaji

    def get_R(self):
        return self.R

class CI_Constraint(Layer):
    def __init__(self, output_dim, return_r = False, epsilon=0.001, momentum=0.99, **kwargs):
        self.output_dim = output_dim
        self.epsilon = epsilon
        self.momentum = momentum
        self.uses_learning_phase = True
        self.return_r = return_r
        super(DecodeGaussianBeta, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.size = input_shape[1] - self.output_dim
        # These are running mean variables, stored for use in test phase.
        self.Vr = self.add_weight(shape=(self.size, self.output_dim),
                                  initializer='zero',
                                  trainable=False,
                                  name='{}_Vr'.format(self.name))
        self.mir = self.add_weight(shape=(1, self.output_dim),
                                   initializer='zero',
                                   trainable=False,
                                   name='{}_mir'.format(self.name))
        self.mjr = self.add_weight(shape=(self.size, 1),
                                   initializer='zero',
                                   trainable=False,
                                   name='{}_mjr'.format(self.name))
        self.vir = self.add_weight(shape=(1, self.output_dim),
                                   initializer='one',
                                   trainable=False,
                                   name='{}_vir'.format(self.name))
        self.vjr = self.add_weight(shape=(self.size, 1),
                                   initializer='one',
                                   trainable=False,
                                   name='{}_vjr'.format(self.name))
        #or just i?
        self.betaj = self.add_weight(shape=(self.size, 1), 
                                   initializer='one',
                                   trainable=True,
                                   name='{}_betaij'.format(self.name)) #self.size
        self.R = K.placeholder(shape = (self.size, 1))
        super(CI_Constraint, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x_cat, mask=None):
        # For some reason, we have to concatenate vectors to feed them using "merge" in keras
        z = x_cat[:, :self.size]
        x = x_cat[:, self.size:]
        batch_size = K.cast(K.shape(x)[0], x.dtype)  # This is a node tensor, so we can't treat as integer
        div_n = Lambda(lambda v: v / batch_size)  # Dividing by batch size is an operation on unknown tensor

        # batch statistics
        mi = K.expand_dims(K.mean(x, axis=0), 0)  # mean of x_i
        mj = K.expand_dims(K.mean(z, axis=0), 1)  # mean of z_j
        vj = K.expand_dims(K.var(z, axis=0) + self.epsilon, 1)  # sigma_j^2
        vi = K.expand_dims(K.var(x, axis=0) + self.epsilon, 0)  # sigma_i^2
        V = div_n(K.dot(K.transpose(z), x))  # j i

        self.add_update([K.moving_average_update(self.Vr, V, self.momentum),
                         K.moving_average_update(self.mir, mi, self.momentum),
                         K.moving_average_update(self.mjr, mj, self.momentum),
                         K.moving_average_update(self.vjr, vj, self.momentum),
                         K.moving_average_update(self.vir, vi, self.momentum)
                         ], x_cat)
        V = K.in_train_phase(V, self.Vr)
        mi = K.in_train_phase(mi, self.mir)
        mj = K.in_train_phase(mj, self.mjr)
        vj = K.in_train_phase(vj, self.vjr)
        vi = K.in_train_phase(vi, self.vir)
        rho = (V - mi * mj) / K.sqrt(vi * vj)
        Q = tf.multiply(rho / (1 - K.square(rho)), self.betaj)
        self.R = K.sum(rho * Q, axis=0, keepdims=True)
        Q = Q / (1 + self.R)
        if self.return_r:
          return self.R
        else:
          return mi + K.sqrt(vi) * K.dot(K.transpose((K.transpose(z) - mj) / K.sqrt(vj)), Q)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_w(self):
        # For visualization
        rho = (self.Vr - self.mir * self.mjr) / K.sqrt(self.vir * self.vjr)
        Q = tf.multiply(self.betaj, rho) / (1 - K.square(rho))
        R = K.sum(rho * Q, axis=0, keepdims=True)
        Q = Q / (1 + R)
        return Q #, self.betaij

    def get_betas(self):
        return self.betaj

    def get_R(self):
        return self.R


#same q dist under p and p*
#p(r,s) #p(r)p(s)q(r|s)^b
#sum p(s)q(r|s)^b


# class DecodeGaussianCD(Layer):
#     """ This decoder assumes a continuous input and a continuous output. It fixes the form of the
#     transformation based on an assumption of conditional independence.
#     It requires access to the true decoded layer at training time. In keras, you need to at it to the input of
#     layer like this:
#     merged_vector = merge([z, x], mode='concat', concat_axis=-1)
#     z = ms.DecodeGaussian(original_dim, name='decoder1')(merged_vector)
#     """
#     def __init__(self, output_dim, epsilon=0.001, momentum=0.99, **kwargs):
#         self.output_dim = output_dim
#         self.epsilon = epsilon
#         self.momentum = momentum
#         self.uses_learning_phase = True
#         super(DecodeGaussianCD, self).__init__(**kwargs)

#     def build(self, input_shape):
#         # Create a trainable weight variable for this layer.
#         self.size = input_shape[1] - self.output_dim
#         # These are running mean variables, stored for use in test phase.
#         self.Vr = self.add_weight(shape=(self.size, self.output_dim),
#                                   initializer='zero',
#                                   trainable=False,
#                                   name='{}_Vr'.format(self.name))
#         self.mir = self.add_weight(shape=(1, self.output_dim),
#                                    initializer='zero',
#                                    trainable=False,
#                                    name='{}_mir'.format(self.name))
#         self.mjr = self.add_weight(shape=(self.size, 1),
#                                    initializer='zero',
#                                    trainable=False,
#                                    name='{}_mjr'.format(self.name))
#         self.vir = self.add_weight(shape=(1, self.output_dim),
#                                    initializer='one',
#                                    trainable=False,
#                                    name='{}_vir'.format(self.name))
#         self.vjr = self.add_weight(shape=(self.size, self.size),
#                                    initializer='one',
#                                    trainable=False,
#                                    name='{}_vjr'.format(self.name))
#         self.vjr_ci = self.add_weight(shape=(self.size, 1),
#                                    initializer='one',
#                                    trainable=False,
#                                    name='{}_vjr'.format(self.name))
#         self.R = K.placeholder(shape = (self.size, 1))
#         super(DecodeGaussianCD, self).build(input_shape)  # Be sure to call this somewhere!

#     def call(self, x_cat, mask=None):
#         # For some reason, we have to concatenate vectors to feed them using "merge" in keras
#         z = x_cat[:, :self.size]
#         x = x_cat[:, self.size:]
#         batch_size = K.cast(K.shape(x)[0], x.dtype)  # This is a node tensor, so we can't treat as integer
#         div_n = Lambda(lambda v: v / batch_size)  # Dividing by batch size is an operation on unknown tensor
#         inv = Lambda(lambda v: matrix_inverse(v))
#         # batch statistics
#         self.mi = K.expand_dims(K.mean(x, axis=0), 0)  # mean of x_i
#         self.mj = K.expand_dims(K.mean(z, axis=0), 1)  # mean of z_j
#         self.vj = div_n(K.dot(K.transpose(z-K.transpose(self.mj)), z-K.transpose(self.mj)))  # sample cov z
#         self.vi = K.expand_dims(K.var(x, axis=0) + self.epsilon, 0)  # sigma_i^2
#         self.vj_ci = K.expand_dims(K.var(z, axis=0) + self.epsilon, 1)  # sigma_j^2

#         self.V = div_n(K.dot(K.transpose(z-K.transpose(self.mj)), x-self.mi)) # j i

#         self.add_update([K.moving_average_update(self.Vr, self.V, self.momentum),
#                          K.moving_average_update(self.mir, self.mi, self.momentum),
#                          K.moving_average_update(self.mjr, self.mj, self.momentum),
#                          K.moving_average_update(self.vjr, self.vj, self.momentum),
#                          K.moving_average_update(self.vir, self.vi, self.momentum),
#                          K.moving_average_update(self.vjr_ci, self.vj_ci, self.momentum),
#                          ], x_cat)

#         V = K.in_train_phase(self.V, self.Vr)
#         mi = K.in_train_phase(self.mi, self.mir)
#         mj = K.in_train_phase(self.mj, self.mjr)
#         Vj = K.in_train_phase(self.vj, self.vjr)
#         vi = K.in_train_phase(self.vi, self.vir)
#         vj_ci = K.in_train_phase(self.vjr_ci, self.vj_ci)

#         #switch statistics to incorporate momentum?
#         # V = K.in_train_phase(self.Vr, self.Vr)
#         # mi = K.in_train_phase(self.mir, self.mir)
#         # mj = K.in_train_phase(self.mjr, self.mjr)
#         # Vj = K.in_train_phase(self.vjr, self.vjr)
#         # vi = K.in_train_phase(self.vir, self.vir)
#         # vj_ci = K.in_train_phase(self.vjr_ci, self.vjr_ci)

#         # CI mean / cov
#         rho = V / K.sqrt(vi*vj_ci)
#         Q = rho / (1 - K.square(rho))
#         self.R = K.sum(rho * Q, axis=0, keepdims=True)
#         Q = Q / (1 + self.R)
#         ci_mu = mi + K.sqrt(vi) * K.dot(K.transpose((K.transpose(z) - mj) / K.sqrt(vj_ci)), Q)
#         ci_var= 1 / (1 + self.R)
#         ci_var = tf.multiply(K.ones_like(ci_mu), ci_var)
#         merged_ci= merge([ci_mu, ci_var], mode='concat', concat_axis=-1)

#         # Full output = mean / cov
#         cond_mu = K.transpose(K.dot(K.transpose(V), inv(Vj)))
#         full_mu = mi + K.dot(z-K.transpose(mj), cond_mu)
#         full_var = vi - diag_part(K.dot(K.transpose(cond_mu), V))
#         full_var = tf.multiply(K.ones_like(full_mu), full_var)
#         merged_full= merge([full_mu, full_var], mode='concat', concat_axis=-1)

#         merged_output = merge([merged_full, merged_ci], mode ='concat', concat_axis=-1)
#         #print K.shape(merged_output)
#         return merged_output

#     def get_output_shape_for(self, input_shape):
#         return (input_shape[0], self.output_dim)

#     def get_w(self):
#         return K.transpose(K.dot(K.transpose(self.Vr), matrix_inverse(self.vjr)))

#     def get_Ri(self):
#         return self.R

#     def get_yj_means(self):
#         return K.transpose(K.in_train_phase(self.mj, self.mjr))

#     def get_yj_vars(self):
#         # update for full
#         return K.transpose(K.in_train_phase(self.vj, self.vjr))

#     def get_prev_layer(self):
#         return Layer
    #x_i = Wij * y_j = Wi * y
    #cov matrix for  h(x_i - g(y)) for minsyn decoder
    #gaussian y, true y, min entropy which pushes down h(x_i |y)






# class InformationDropout(Layer):
#     """
#         Parameters:
#         -----------
#         batch_size: Both Keras backends need the batch_size to be defined before
#             hand for sampling random numbers. Make sure your batch size is kept
#             fixed during training. You can use any batch size for testing.
#         regularizer_scale: By default the regularization is already proberly
#             scaled if you use binary or categorical crossentropy cost functions.
#             In most cases this regularizers should be kept fixed at one.
#     """
#     def __init__(self, output_dim, activation='relu',
#                  q0 = .1, **kwargs):
#         self.q0 = q0
#         self.uses_learning_phase = True
#         #self.batch_size = batch_size
#         #self.init = initializations.get(init)
#         self.activation = activations.get(activation)
#         self.output_dim = output_dim
#         #self.initial_weights = weights
#         #self.input_dim = input_dim
#         #if self.input_dim:
#         #    kwargs['input_shape'] = (self.input_dim,)
#         #self.input = K.placeholder(ndim=2)
#         super(InformationDropout, self).__init__(**kwargs)


#     def build(self, input_shape):
#         # Create a trainable weight variable for this layer.
#         if input_shape[0] is None:
#           self.ns = 1
#         else:
#           self.ns = input_shape[0]
        
#         self.size = input_shape[1]
#         #self.W_mean = self.add_weight(shape=(self.size, self.output_dim),
#         #                         initializer='normal',
#         #                         trainable=True,
#         #                         name='{}_W'.format(self.name))
#         #self.b_mean = K.zeros((self.output_dim,))

#         self.W_noise = self.add_weight(shape=(self.size, self.output_dim),
#                                  initializer='normal',
#                                  trainable=True,
#                                  name='{}_W'.format(self.name))

#         # OR JUST SET Q0 = CONSTANT
#         #self.q0 = K.zeros((self.output_dim,))


#         #self.beta = self.add_weight(shape=(self.output_dim,),
#         #                            initializer='zero',
#         #                            trainable=True,
#         #                            name='{}_beta'.format(self.name))
#         #self.gamma = self.add_weight(shape=(self.output_dim,),
#         #                            initializer='one',
#         #                            trainable=False,
#         #                            name='{}_gamma'.format(self.name))
#         self.trainable_weights = [self.W_noise]#, self.q0]

#         #self.regularizers = []
#         #reg = self.get_kl(self.get_input())
#         #self.regularizers.append(reg)
        
#         super(InformationDropout, self).build(input_shape)  # Be sure to call this somewhere!

#     def call(self, X, mask=None):
#         #where the logic lives!!!
#         noise_std = self.activation(K.dot(X, self.W_noise)) #self.ns x self.output_dim
#         #batch, features = shape(noise_std)  
#         #if batch is None or features is None:
#         #  batch = 0
#         #  features = 0
#         #self.noise = K.zeros_like(noise_std) #np.zeros((batch, features))
#         self.noise = K.ones((self.ns, self.size))
#         batch, features = self.noise.get_shape()
#         print batch, features
#         if isinstance(batch, int):
#           for i in range(batch):
#             for j in range(features):
#               if X[i,j] == 0:
#                 self.noise[i, j] = self.q0
#               else:
#                 self.noise[i, j] = np.log(np.random_normal(0, noise_std[i,j]**2, 1))
#         self.noise = K.variable(value = self.noise)
#         return self.noise

#     def get_output_shape_for(self, input_shape):
#         return (input_shape[0], self.output_dim)

#     def get_noise(self):
#         return self.noise

#     def _get_output(self, X, train=False):
#         noise_std = self.activation(K.dot(X, self.W_noise)) #self.ns x self.output_dim
#         if train:
#             batch, features = shape(noise_std)  
#             if batch is None or features is None:
#               batch = 0
#               features = 0
#             self.noise = np.zeros((batch, features))
#             for i in range(batch):
#               for j in range(features):
#                 if X[i,j] == 0:
#                   self.noise[i, j] = self.q0
#                 else:
#                   self.noise[i, j] = np.log(np.random_normal(0, noise_std[i,j]**2, 1))
#             self.noise = K.variable(value = self.noise)
#             return self.noise
#         else:
#             return mean

#     def get_output(self, train=False):
#         X = self.get_input()
#         return self._get_output(X, train)

#     @property
#     def output_shape(self):
#         return (self.input_shape[0], self.output_dim)


# 