# Tensorflow Template

A template for running and managing tensorflow experiments. Features are 
* Save the model, and restore from previous training checkpoints
* Run inference with trained model by loading its saved configurations

### Network
Define network, graph and provide train step.

build network 
```
    def build(self):
        print ("Start building Network")
        self.build_placeholder()
        self.build_network()
        print ("Done.")
```

The interfaces to training and inference are
```
    def train_step(self, sess, input_batch, target_batch)
    def inference(self, sess, input_batch)
```

### Training
Main file for train and save out trained model. All traning parameters are flags in train.py which will be saved in `config.json` in model folder.

### For inference
Restore from log file.

