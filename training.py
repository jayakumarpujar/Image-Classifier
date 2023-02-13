@jayakumarpujar

import sys
import argparse
import logging
import datetime
import os
import shutil
import glob
import csv
import ntpath
import time
import pandas as pd
import yaml
import keras
import tensorflow as tf
import efficientnet.tfkeras
import numpy as np
from  efficientnet.tfkeras import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from PIL import ImageFile
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
from architectures import ArchitectureSearch

start_time = time.time()


# from tensorflow.keras.models import Sequential
# from tensorflow.keras.utils import multi_gpu_model

ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Hyperparameters
with open('hyperparameter.yaml') as file:
    hyperparameters = yaml.load(file, Loader = yaml.FullLoader)
print('==================================================================================================')
for p in hyperparameters:
    print(str(p) + " = " + str(hyperparameters[p]))
print('==================================================================================================')

num_classes = hyperparameters['num_classes']
project_name = hyperparameters['project_name']
data_set_name = hyperparameters['data_set_name']
train_data_dir = '/data/' + str(project_name) + '/' + str(data_set_name) + '/training/train'
valid_data_dir = '/data/' + str(project_name) + '/' + str(data_set_name) + '/training/validation'
evaluation_data_dir = '/data/' + str(project_name) + '/' + str(data_set_name) + '/evaluation/*/*'
input_shape = hyperparameters['input_shape']
architecture = hyperparameters['architecture']
class_mode = hyperparameters['class_mode']
optimizer = hyperparameters['optimizer']
loss = hyperparameters['loss']
epochs = hyperparameters['epochs']
batch_size = hyperparameters['batch_size']
learning_rate = hyperparameters['learning_rate']
gpus = hyperparameters['gpus']
path_to_save_final_model = 'imageclassification/classifier-models/' + str(project_name) + '/'
logdir = path_to_save_final_model
patience = hyperparameters['patience']
sub_architecture_class = hyperparameters['sub_architecture_class']



train_time = str(datetime.datetime.now())
train_time = train_time.replace(".", "_")
train_time = train_time.replace(":", "_")
train_time = train_time.replace(" ", "_")



class TrainAndTest(Callback):
    ''' Training and Testing class '''
    def __init__(self):
        super(TrainAndTest, self).__init__()
        self.train_data_dir = train_data_dir
        self.valid_data_dir = valid_data_dir
        self.evaluation_data_dir = evaluation_data_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.class_mode = class_mode
        self.input_shape = input_shape
        self.optimizer = optimizer
        self.loss = loss
        self.architecture = architecture
        self.path_to_save_final_model = path_to_save_final_model
        self.patience = patience
        self.logdir = logdir
        self.train_time = train_time
        self.gpus = gpus
        self.sub_architecture_class = sub_architecture_class
        self.num_classes = num_classes

    def train(self):
        ''' training function'''
        if str(self.gpus) == '0':
            strategy = tf.distribute.MirroredStrategy(["GPU:0"])
        elif str(self.gpus) == '0,1':
            strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
        elif str(self.gpus) == '0,1,2,3':
            strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1", "GPU:2", "GPU:3"])

        with strategy.scope():

        # model architecture
            architecture_time = time.time()
            if str(self.sub_architecture_class) == 'applications':
                class_name = ArchitectureSearch()
                model = class_name.tensorflow_applications()
                print("Using tensorflow.applications")
            elif str(self.sub_architecture_class) == 'efficientnet':
                class_name = ArchitectureSearch()
                model = class_name.sota()
                print("Using efficientnet.tfkeras module instead from tensorflow applications")
            elif str(self.sub_architecture_class) == 'none':
                model = tf.keras.models.load_model(self.architecture)
            print("--------Model or architecture loaded-------------")
            print("architecture_loading_time ", (time.time()-architecture_time)/60, " mnts")

            datagen_time = time.time()
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)
            print("datagenerator time ", (time.time()-datagen_time)/60, " mnts")
            
            if self.num_classes <= 2:
                loss = "binary_crossentropy"
                class_mode = "binary"
            elif self.num_classes > 2:
                loss = "categorical_crossentropy"
                class_mode = "categorical"
        # Data Generator
            train_gen_time = time.time()
            train_generator = datagen.flow_from_directory(
                directory = str(self.train_data_dir),
                target_size = (self.input_shape, self.input_shape),
                classes = ['real','fake'],
                class_mode = class_mode,
                batch_size = self.batch_size,
                interpolation = "lanczos")
            print("train_gen_time ", (time.time()-train_gen_time)/60, " mnts")
            print("train_gen_type",type(train_generator),train_generator)

            val_gen_time = time.time()
            validation_generator = datagen.flow_from_directory(
                directory = str(self.valid_data_dir),
                target_size = (self.input_shape, self.input_shape),
                classes = ['real','fake'],
                class_mode = class_mode,
                batch_size = self.batch_size,
                interpolation = "lanczos")
            print("val_gen_time ", (time.time()-val_gen_time)/60, " mnts")
            print("val_gen_type",type(validation_generator),validation_generator)



            # optimiser function
            optimizer_time = time.time()
            optimiser = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)
            print("optimizer time ", (time.time()-optimizer_time)/60, " mnts")

            compilation_time = time.time()
            model.compile(loss = loss, optimizer = optimiser, metrics = ['accuracy'])
            print("compilation_time ", (time.time()-compilation_time)/60, " mnts")


            checkpoints_filepath = os.path.join(str(self.path_to_save_final_model) + str(self.train_time) + '/' + 'model_{epoch:02d}_{val_accuracy:.03f}.h5')

            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                checkpoints_filepath,
                monitor = 'val_accuracy',
                verbose = 1,
                save_best_only = True,
                mode = 'max')

            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                min_delta = 0,
                patience = self.patience,
                verbose = 1,
                mode = 'max',
                baseline = None,
                restore_best_weights = True)

            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = os.path.join(self.logdir + str(train_time)))

            callbacks_list = [checkpoint, early_stopping, tensorboard_callback]
            if str(self.gpus) == "0" or str(self.gpus) == "1":
                workers = 6
            elif str(self.gpus) == "0,1":
                workers = 12
            elif str(self.gpus) == "0,1,2,3":
                workers = 24

            os.system(GPU_COMMAND)
            training_time = time.time()
            model.fit(
                train_generator,
                batch_size = self.batch_size,
                epochs = self.epochs,
                verbose = 2,
                validation_data = validation_generator,
                callbacks = callbacks_list,
                workers = workers,
                shuffle = True)
            tf.keras.models.save_model(model, str(self.path_to_save_final_model) + str(self.train_time) +'/final_epoch.h5')
            os.system(GPU_COMMAND)
            print("training time ", (time.time()-training_time)/60, " mnts")

    def test(self):

        ''' testing function '''
        os.system(GPU_COMMAND)
        def path_leaf(path):
            head, tail = ntpath.split(path)
            return tail or ntpath.basename(head)

        print("Evaluating the model's >>>>>>")
        def preprocessing_img(img_path):
            img = tf.keras.preprocessing.image.load_img(img_path, target_size = (self.input_shape, self.input_shape), interpolation = "lanczos")
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis = 0)
            img_array /= 255.
            return img_array

        #path to models
        models = glob.glob(str(self.path_to_save_final_model) + str(train_time) + "/*.h5")

        for model in models:
            name = model.split("/")[-1].split(".h5")[0]
            print("Model Name: ", name)
            model = tf.keras.models.load_model(model)
            imgs_dir = glob.glob(self.evaluation_data_dir)
            print(imgs_dir)
            output_file = open(str(self.path_to_save_final_model) + str(train_time) + '/' + str(name) + '.csv', 'w', newline = '')
            if self.num_classes <= 2:
                row = ['image_name', 'pred_score' , 'GT']
            elif self.num_classes == 3:
                row = ['image_name', 'class_1', 'class_2', 'class_3', 'GT']
            elif self.num_classes == 4:
                row = ['image_name', 'class_1', 'class_2', 'class_3', 'class_4', 'GT']
            elif self.num_classes == 5:
                row = ['image_name', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'GT']
            elif self.num_classes == 6:
                row = ['image_name', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6', 'GT']
            elif self.num_classes == 7:
                row = ['image_name', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6', 'class_7', 'GT']
            elif self.num_classes == 8:
                row = ['image_name', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6', 'class_7', 'class_8', 'GT']
            elif self.num_classes == 9:
                row = ['image_name', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6', 'class_7', 'class_8', 'class_9', 'GT']
            elif self.num_classes == 10:
                row = ['image_name', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6', 'class_7', 'class_8', 'class_9', 'class_10', 'GT']

            csv_writer  = csv.writer(output_file)
            csv_writer.writerow(row)
            count = 0
            for image_path in imgs_dir[:]:

                ground_truth = image_path.split("/")[-2]
                filename = path_leaf(image_path)
                preproc_image = preprocessing_img(image_path)
                model_pred = model.predict(preproc_image)
                count += 1
                if count % 100 == 0:
                    print("completed >>>> ", count)

                if self.num_classes <= 2:
                    row = [filename, model_pred[0][0], ground_truth]
                elif self.num_classes == 3:
                    row = [filename, model_pred[0][0], model_pred[0][1], model_pred[0][2], ground_truth]
                elif self.num_classes == 4:
                    row = [filename, model_pred[0][0], model_pred[0][1], model_pred[0][2], model_pred[0][3], ground_truth]
                elif self.num_classes == 5:
                    row = [filename, model_pred[0][0], model_pred[0][1], model_pred[0][2], model_pred[0][3], model_pred[0][4], ground_truth]
                elif self.num_classes == 6:
                    row = [filename, model_pred[0][0], model_pred[0][1], model_pred[0][2], model_pred[0][3], model_pred[0][4], model_pred[0][5], ground_truth]
                elif self.num_classes == 7:
                    row = [filename, model_pred[0][0], model_pred[0][1], model_pred[0][2], model_pred[0][3], model_pred[0][4], model_pred[0][5], model_pred[0][6], ground_truth]
                elif self.num_classes == 8:
                    row = [filename, model_pred[0][0], model_pred[0][1], model_pred[0][2], model_pred[0][3], model_pred[0][4], model_pred[0][5], model_pred[0][6], model_pred[0][7], ground_truth]
                elif self.num_classes == 9:
                    row = [filename, model_pred[0][0], model_pred[0][1], model_pred[0][2], model_pred[0][3], model_pred[0][4], model_pred[0][5], model_pred[0][6], model_pred[0][7], model_pred[0][8], ground_truth]
                elif self.num_classes == 10:
                    row = [filename, model_pred[0][0], model_pred[0][1], model_pred[0][2], model_pred[0][3], model_pred[0][4], model_pred[0][5], model_pred[0][6], model_pred[0][7], model_pred[0][8], model_pred[0][9], ground_truth]
                csv_writer.writerow(row)

            print(name, 'tested')
    def model_register(self):
        ''' model register function '''
        shutil.make_archive((str(self.path_to_save_final_model) + str(self.train_time)), 'zip', self.path_to_save_final_model, str(self.train_time))
        for lst in os.listdir(str(self.path_to_save_final_model)):
            if lst == str(self.train_time) + ".zip":
                upload_file_path_main = str(self.path_to_save_final_model) + str(self.train_time) + ".zip"
                print("upload_file_path_main :", upload_file_path_main)
                blob = bucket.blob(upload_file_path_main)
                blob.upload_from_filename(upload_file_path_main)
    print("Total time taken ", (time.time()-start_time)/60, " mnts")
