"""Execute inference only on trained model
"""
import json
import os, sys
import numpy as np
import tensorflow as tf

from model import LinearModel 
from tensorflow.examples.tutorials.mnist import input_data

tf.app.flags.DEFINE_string("model_path", "logs/linear-mnist", "Path to the trained model")
tf.app.flags.DEFINE_string("data_path", "dataset/MNIST", "Path to the dataset folder")
tf.app.flags.DEFINE_string("outname", "None", "Name of output CSV file")

FLAGS = tf.app.flags.FLAGS

def load_model(sess, saved_path):
    class Config:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    
    # Load configuration from trained model
    with open("/".join([saved_path, "config.json"]), "r") as file:
        saved = json.load(file)
    print ("Configuration recoverd:")
    pprint (saved)
    config = Config(**saved)
    
    if config.model == "vallina":
        model = LinearModel(config, "inference")
        model.build()
    else:
        pass
        # other model

    print ("Restore from previous results...")
    ckpt = tf.train.get_checkpoint_state(saved_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print ("saved path: {}".format(saved_path))
        model.restore(sess, saved_path)
        print ("Model reloaded from {}".format(ckpt.model_checkpoint_path))
    else:
        raise ValueError("CAN NOT FIND CHECKPOINTS EXISTS!")
    return model

def inference(sess, test_data, model):
    """Used to predict results on test dataset
    """
    # Prepare path to output
    OUTS=[]
    for idx, datum in enumerate(test_data):
        # Predict one datum a time
        datum = np.expand_dims(datum, axis=0)
        predicted_categories = model.inference(sess, datum)
        OUTS.append(predicted_categories)
    return OUTS

def evaluation(results, targets):
    """Calculate accuracy with labels and predicted classes
    """
    accuracy = 0
    for yhat, y in zip(results, targets):
        if yhat == y:
            accuracy += 1
    accuracy /= len(results)
    return accuracy

def main():
    gpu_options = tf.GPUOptions(allow_growth=True)

    print ("Start loading test dataset...")
    mnist = input_data.read_data_sets(FLAGS.data_path, one_hot=False)
    test_data = mnist.test.images
    targets = mnist.test.labels

    print ("Loading is done.")


    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        print ("Loading the trained model...")
        model = load_model(sess, FLAGS.model_path)
        print ("Done.")

        print ("Predicting ...")
        results = inference(sess, test_data, model)
        print ("Done.")

        acc = evaluation(results, targets)
        print ("Accuracy of on testing dataset: {}".format(acc))

if __name__ == "__main__":
    main()
