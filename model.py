"""Define Computational Graphs
"""
import tensorflow as tf
import tensorflow.contrib.layers as layers


class LinearModel(object):
    def __init__ (self, config, mode):
        assert mode.lower() in ["train", "inference"]
        self.my_name = "linear_model"
        self.mode = mode.lower()
        self.config = config

        self.feat_dim = self.config.feat_dim
        self.num_categories = self.config.num_categories
        self.reg = self.config.reg

        # Initializer
        if self.config.init_method == "zeros":
            self.init_method=None
        elif self.config.init_method == "xavier":
            self.init_method = layers.xavier_initializer()
        elif self.config.init_method == "ortho":
            self.init_method = tf.orthogonal_initializer()

        if self.mode == "train":
            self.optimizer_type = self.config.optimizer_type
            self.learning_rate = self.config.learning_rate
            self.clip_grad = self.config.clip_grad
            self.global_step = tf.Variable(0, trainable=False, name='global_step')

    def build(self):
        print ("Start building Network")
        self.build_placeholder()
        self.build_network()
        print ("Done.")

    def build_placeholder(self):
        print("Building placeholders...")
        with tf.variable_scope("Input"):
            self.input = tf.placeholder(dtype=tf.float32, 
                                shape=(None, self.feat_dim), name="input")
            self.label = tf.placeholder(dtype=tf.int32, shape=(None,), name="label")
    
    def build_network(self):
        print ("Building network...")
        with tf.name_scope("YOUR_NETWROK") as scope:

            with tf.variable_scope("LinearModel"):

                w = tf.get_variable("w", shape=[self.feat_dim,  self.num_categories], 
                                    initializer=tf.random_normal_initializer(),
                                    regularizer=layers.l2_regularizer(self.reg))

                b = tf.get_variable("b", shape=[self.num_categories], 
                                        initializer=tf.constant_initializer(0.01),
                                        regularizer=layers.l2_regularizer(self.reg))

                linout = tf.matmul(self.input, w) + b
                self.logits = linout
                self.probs = tf.nn.softmax(self.logits)
                self.preds = tf.argmax(input=self.probs, axis=1)

        # If train mode, define loss function
        if self.mode == "train":
            print ("Define Loss function")
            with tf.variable_scope("Loss"):
                onehot_labels = tf.one_hot(indices=tf.cast(self.label, tf.int32), depth=self.num_categories)
                self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=self.logits))
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.preds, tf.argmax(onehot_labels, 1)), tf.float32))

            # initialize optimizer in train mode
            self.init_optimizer()

            # set summary operator in train mode with layman method
            tf.contrib.layers.summarize_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("accuracy", self.accuracy)
            self.summary_op = tf.summary.merge_all()

    def init_optimizer(self):
        """Automatically initialize when using train mode.
            TODO: learning rate decay
        """
        if self.mode == "train":
            print ("Initialize Optimizer ({})".format(self.optimizer_type))
            train_params = tf.trainable_variables()

            with tf.variable_scope("Solver"):
                if self.optimizer_type.lower() == "sgd":
                    self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
                elif self.optimizer_type.lower() == "rmsprop":
                    self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
                elif self.optimizer_type.lower() == "adam":
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                elif self.optimizer_type.lower() == "adagrad":
                    self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate)

                gradients = tf.gradients(self.loss, train_params)
                # clip
                clip_gradients, _ = tf.clip_by_global_norm(gradients, self.clip_grad)
                # update
                self.updates = self.optimizer.apply_gradients(
                    zip(clip_gradients, train_params), global_step=self.global_step)

    def train_step(self, sess, input_batch, target_batch):
        if self.mode.lower() != 'train':
            raise ValueError ('Train step function can only used in train mode.')
        input_feed = {self.input: input_batch, 
                        self.label: target_batch}
        output_feed = [self.updates, self.loss]
        outs = sess.run(output_feed, input_feed)
        return outs[1]

    def eval_step(self, sess, input_batch, target_batch):
        # these data should come from test samples
        input_feed = {self.input: input_batch,
                        self.label: target_batch}
        output_feed = [self.loss, self.accuracy, self.summary_op]
        outputs = sess.run(output_feed, input_feed)
        return outputs

    def inference(self, sess, input_batch):
        if self.mode != "inference":
            raise ValueError("Inference function can only be used in inference mode.")
        input_feed = {self.input: input_batch}
        output_feed = [self.preds]
        outputs = sess.run(output_feed, input_feed)
        return outputs[0]

    def restore(self, sess, path):
        checkpoint = tf.train.get_checkpoint_state(path)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print ("Model restore from {}".format(checkpoint.model_checkpoint_path))
        else:
            raise ValueError('Can not load from checkpoints')

    def save(self, sess, path, var_list=None, global_step=None):
        saver = tf.train.Saver()
        path = path + "/" + self.my_name
        save_path = saver.save(sess, path, global_step)
        print ('model (global steps: {}) is saved at {}'.format(global_step.eval(), save_path))

    @property
    def vars(self):
        """Return all global variables
        """
        return [var for var in tf.global_variables() if self.name in var.name]