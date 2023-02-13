@jayakumarpujar

import yaml
import tensorflow as tf
import efficientnet.tfkeras 

with open('hyperparameter.yaml') as file:
    hyperparameters = yaml.load(file, Loader = yaml.FullLoader)

input_shape = hyperparameters['input_shape']
image_channels = hyperparameters['image_channels']
include_top = hyperparameters['include_top']
num_classes = hyperparameters['num_classes']
sub_architecture_class = hyperparameters['sub_architecture_class']
architecture = hyperparameters['architecture']

class ArchitectureSearch:
    ''' architecture class '''
    def __init__(self):

        self.input_shape = input_shape
        self.image_channels = image_channels
        self.include_top = include_top
        self.num_classes = num_classes
        self.sub_architecture_class = sub_architecture_class
        self.architecture = architecture

    def tensorflow_applications(self):
        ''' tf.keras.applications '''
        input_tensor = tf.keras.layers.Input(shape = (self.input_shape, self.input_shape, self.image_channels))
        base_model = tf.keras.__dict__[self.sub_architecture_class].__dict__[self.architecture](
            input_shape = None, 
            alpha = 1.0, 
            include_top = self.include_top, 
            pooling = 'avg', 
            weights = "imagenet")(input_tensor)
        if self.include_top == True:
            model = base_model
        else:
            out1 = tf.keras.layers.Dense(8,activation = "relu", name = 'dense_01')(base_model)
            out2 = tf.keras.layers.Dense(4,activation = "relu", name = 'dense_02')(out1)
            if self.num_classes > 2 :
                output = tf.keras.layers.Dense(self.num_classes, activation = "softmax", name = 'dense_03')(out2)
            else:
                output = tf.keras.layers.Dense(1, activation = "sigmoid", name = 'dense_03')(out2)
            model = tf.keras.Model(inputs = [input_tensor], outputs = [output], name = 'mobilenet')
        return model


    def sota(self):
        ''' efficientnet.tfkeras'''
        input_tensor = tf.keras.layers.Input(shape = (self.input_shape, self.input_shape, self.image_channels))
        base_model = efficientnet.tfkeras.__dict__[self.architecture](include_top = self.include_top, 
                                                                weights = "imagenet",
                                                                input_tensor = input_tensor)
        if self.include_top == True:
            model = base_model    
        else: 
            global_avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
            dense_1 = tf.keras.layers.Dense(20, activation = "relu")(global_avg)
            dropout_2 = tf.keras.layers.Dropout(0.3)(dense_1)
            dense_2 = tf.keras.layers.Dense(10, activation = "relu")(dropout_2)
            if self.num_classes > 2:
                dense_3 = tf.keras.layers.Dense(self.num_classes, activation = "softmax")(dense_2)
            else:
                dense_3 = tf.keras.layers.Dense(1, activation = "sigmoid")(dense_2)
            model = tf.keras.models.Model(inputs = base_model.inputs, outputs = dense_3)
        return model


if __name__ == "__main__":
    architecture_search()
