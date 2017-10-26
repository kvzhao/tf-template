"""For training task
"""
import os, sys
import json
import shutil
from tqdm import tqdm
from pprint import pprint
import numpy as np
import tensorflow as tf

from model import LinearModel 

from tensorflow.examples.tutorials.mnist import input_data

print ("Version of tensorflow is {}".format(tf.__version__))

# General
tf.app.flags.DEFINE_bool("is_train", True, "Option for mode. Set False to run inference. (and output submit file?)")
tf.app.flags.DEFINE_string("model", "vallina", "Choose the type of model.")
tf.app.flags.DEFINE_string("logdir", "logs", "Name of output folder")
tf.app.flags.DEFINE_string("task_name", "linear-mnist", "Name of this training task")
tf.app.flags.DEFINE_bool("reset", False, "Training start from stratch")

# Data path
tf.app.flags.DEFINE_string("data_path", "dataset/MNIST", "Path to dataset.")

# Network
tf.app.flags.DEFINE_integer("feat_dim", 784, "Image size of mnist is 28*28 = 784")
tf.app.flags.DEFINE_integer("num_categories", 10, "10 classes in MINIT Dataset.")
tf.app.flags.DEFINE_string("init_method", "zeros", "Methods of initialize weights: zeros/ xavier/ ortho")
tf.app.flags.DEFINE_float("reg", 0.0001, "Regularization of network.")

# Training
tf.app.flags.DEFINE_string("optimizer_type", "ADAM", "Optimizer can be choosen as SGD/ADAM/RMSProp")
tf.app.flags.DEFINE_float("learning_rate", 0.0001, "Initial learning rate")
tf.app.flags.DEFINE_float("clip_grad", 5.0, "Maximum of gradient norm")

tf.app.flags.DEFINE_integer("max_epochs", 10, "Max epoch of training process")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size")
tf.app.flags.DEFINE_integer("val_batch_size", 512, "Batch size for Validation")

tf.app.flags.DEFINE_integer("save_epoch", 5, "Save model per # of epochs")
tf.app.flags.DEFINE_integer("display_freq", 100, "Print step loss per # of steps")
tf.app.flags.DEFINE_integer("eval_freq", 500, "Evaluate model per # of steps")

FLAGS = tf.app.flags.FLAGS

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

def train(sess, train_data, val_data, model, FLAGS):
    # Training Process Parameters
    num_train_data = train_data.num_examples
    num_val_data = val_data.num_examples
    total_batch = num_train_data // FLAGS.batch_size

    # Path
    model_path = os.path.join(FLAGS.logdir, FLAGS.task_name)
    val_log_writer = tf.summary.FileWriter(model_path + '/val', sess.graph)
    train_log_writer = tf.summary.FileWriter(model_path + '/train', sess.graph)

    print ("Start Training...")
    train_loss_hist = []
    eval_loss_hist = []

    for epoch_idx in tqdm(range(FLAGS.max_epochs)):

        epoch_loss = 0.0

        for local_step in range(total_batch):
            # global step
            step = model.global_step.eval()

            batch_seqs, batch_labs = train_data.next_batch(FLAGS.batch_size)

            step_loss, train_accuracy, train_summary = model.train_step(sess, batch_seqs, batch_labs)
            epoch_loss += step_loss

            ## ray try
            train_loss_accumulate = np.append(train_loss_accumulate, step_loss)
            train_accuracy_accumulate = np.append(train_accuracy_accumulate, train_accuracy)

            # Display loss information
            if step % FLAGS.display_freq == 0:
                train_log_writer.add_summary(train_summary, model.global_step.eval())
                print ("Step [ {} ]: training loss:   {:.3f}  |  training accuracy:   {:.3f}".format(
                        step, float(np.mean(train_loss_accumulate)), float(np.mean(train_accuracy_accumulate))))

            # Evaluation (Validation)
            if step % FLAGS.eval_freq == 0:
                val_seqs, val_labs = val_data.next_batch(FLAGS.val_batch_size)
                val_seqlens = np.array([feat_dim] * FLAGS.val_batch_size, dtype=np.int32)
                eval_loss, accuracy, val_summary = model.eval_step(sess, val_seqs, val_labs)
                val_log_writer.add_summary(val_summary, model.global_step.eval())
                print ("Step [ {} ]: validation loss: {:.3f}  |  validation accuracy: {:.3f}".format(step, eval_loss, accuracy))
            
        # End of step loop --- 

        # Save the model
        if epoch_idx % FLAGS.save_epoch == 0:
            model.save(sess, model_path, global_step=model.global_step)

    model.save(sess, model_path, global_step=model.global_step)
    print ("Save the lastest model of training.")

def main():
    # For GPU memory efficiency
    gpu_options = tf.GPUOptions(allow_growth=True)

    # create log dir
    if not os.path.exists(FLAGS.logdir):
        os.makedirs(FLAGS.logdir)
    model_path = os.path.join(FLAGS.logdir, FLAGS.task_name)

    # Train or Inference
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        if FLAGS.is_train: 
            # load traning dataset
            print ("Start Loding Dataset...")
            mnist = input_data.read_data_sets(FLAGS.data_path ,one_hot=False, validation_size=5000)
            train_data = mnist.train
            val_data = mnist.validation
            print ("Loading done.")

            print ("Initialize the network")
            if not os.path.exists(model_path) or FLAGS.reset:
                # Create model if not exist or reset the model.
                model = create_model(sess, FLAGS, mode="train")
            else:
                model = load_model(sess, model_path, mode="train")
                # then we restore the trained model.

            train(sess, train_data, val_data, model, FLAGS)
            print ("Training Done.")
        else:
            print ("The inference mode is removed from inference.py")

if __name__ == "__main__":
    main()