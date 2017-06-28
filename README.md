# disentanglement

This code allows us to create NN architectures using Keras and apply various information theoretic regularizations which may encourage 'disentangled' or 'non-synergistic' representations.

MODEL OPTIONS are specified using models.py:

-Args (training parameters such as epochs, batch_size, optimizer) (fed thru command line in my example)


-EncoderArgs (encoder architecture / loss)

  e.g. e = EncoderArgs(latent_dim, info_dropout = True, activation = 'softplus')

-DecoderArgs (decoder architecture / loss)

  e.g. d = DecoderArgs(minsyn = 'binary', ci_reg = True, activation = 'softplus', initializer = 'orthogonal')
       d = DecoderArgs(reversed(latent_dim[:-1]), screening = True) (final layer of original dimension automatically added)

These objects are fed to SuperModel, which sets the architecture, fits, and runs visualization. See run.py for how to call models, along with some example regularization parameters


LOSS FUNCTION OPTIONS in losses.py:

Reconstruction:  ('recon' argument in SuperModel call)
-any Keras objective (e.g. objectives.binary_crossentropy)
 
-losses.error_entropy* = Gaussian estimator h(xi - g(y)) + non-gaussianity (also used in screening loss)

Encoder Regularization:

-ci_reg (remember to specify minsyn = 'gaussian' or 'binary', gaussian used by default to match continuous NN activations)

-information_dropout

Decoder Regularization:

-minsyn decoder

-ci_reg (remember to specify minsyn = 'gaussian' or 'binary', binary used by default to match pixel output)

-screening

-information_dropout*

-ci_wms (WMS / -TC under the CI decoder)*

*not fully tested


Dependencies (all for 3-letter-MNIST, all but enchant are for obtaining the dataset) 
enchant 
urllib
shutil
struct
gzip
zipfile
