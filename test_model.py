"""Test program for building the network
"""
import tensorflow as tf
from model import LinearModel 

def create_model(sess, FLAGS, mode):
    model = LinearModel(FLAGS, mode)
    model.build()

    # initialize variables
    sess.run(tf.global_variables_initializer())

    return model

def print_newtork(name):
    print ('{}:'.format(name))
    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name):
        print ('\t{}\t{}'.format(var.name, var.get_shape()))

# Network
tf.app.flags.DEFINE_integer("feat_dim", 784, "Image size of mnist is 28*28 = 784")
tf.app.flags.DEFINE_integer("num_categories", 10, "10 classes in MINIT Dataset.")
tf.app.flags.DEFINE_string("init_method", "zeros", "Methods of initialize weights: zeros/ xavier/ ortho")
tf.app.flags.DEFINE_float("reg", 0.0001, "Regularization of network.")
# Training
tf.app.flags.DEFINE_string("optimizer_type", "ADAM", "Optimizer can be choosen as SGD/ADAM/RMSProp")
tf.app.flags.DEFINE_float("learning_rate", 0.0001, "Initial learning rate")
tf.app.flags.DEFINE_float("clip_grad", 5.0, "Maximum of gradient norm")
FLAGS = tf.app.flags.FLAGS

with tf.Session() as sess:
    model = create_model(sess, FLAGS, "train")
    print_newtork(name="LinearModel")
    print ("Test Done.")