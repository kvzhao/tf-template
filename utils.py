"""Utility
    create_model: 
    load_model:
"""
import json
import os, sys
import shutil
from pprint import pprint
import tensorflow as tf

def create_model(sess, FLAGS, mode):
    """Create model only used for train mode.
    """
    if FLAGS.model == "vallina":
        model = LinearModel(FLAGS, mode)
        model.build()
    else:
        pass
        # other model 

    # create task file
    model_path = os.path.join(FLAGS.logdir, FLAGS.task_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        print ("Save model to {}".format(model_path))
    elif (FLAGS.reset):
        shutil.rmtree(model_path)
        os.makedirs(model_path)
        print ("Remove existing model at {} and restart.".format(model_path))
    else:
        raise ValueError("Fail to create the new model.")

    # Save the current configurations
    config = dict(FLAGS.__flags.items())
    with open("/".join([model_path, "config.json"]), "w") as file:
        json.dump(config, file)

    # initialize variables
    sess.run(tf.global_variables_initializer())

    return model

def load_model(sess, saved_path, mode):
    class Config:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    
    # Load configuration from trained model
    with open("/".join([saved_path, "config.json"]), "r") as file:
        saved = json.load(file)
    print ("Configs recoverd...")
    pprint (saved)
    config = Config(**saved)
    
    if config.model == "vallina":
        model = LinearModel(config, mode)
        model.build()
    else:
        pass
        # other model

    print ("Restore from previous results...")
    ckpt = tf.train.get_checkpoint_state(saved_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print ("from saved path: {}".format(saved_path))
        model.restore(sess, saved_path)
    else:
        raise ValueError("CAN NOT FIND CHECKPOINTS EXISTS!")
    return model