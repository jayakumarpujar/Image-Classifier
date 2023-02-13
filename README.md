# Image-Classifier
Binary or Multi Classifier to classify images by using Deep learning Architecture. 

To train the model with mobilenet architectures from tensorflow.keras.applications then change the values in yaml file (parameter.yaml)


- if MObileNet / MobileNetv2 / EfficientNetB0 to EfficientNetB7 from tensorflow.keras.applications then:
sub_architecture_class as 'applications' and architecture as required architecture name (example - EfficientNetB0 or MobileNet )

- if EfficientNetB0 to B7 from efficientnet.tfkeras then:
sub_architecture_class as 'efficientnet' and architecture as 'EfficientNetB0 to EfficientNetB7' (choose any one architecture from the list)

- if retraining is a choice then create a one diectory with in the classification-mlops and upload the base model within and copy & paste the path at
architecture and 'none' at sub_architecture_class

To train multiclassifier model
- choose number of classes and update the value of num_classes in yaml file (example - if choosen three classes then mention it as 3)

- final_dense_activation as softmax
- class_mode as categorical
- loss as categorical_crossentropy
- csv classing format will be in alphabetical order of your classes

To train binary classifier model
- choose num_classes as 2
- final_dense_activation as sigmoid
- class_mode as binary
- loss as binary_crossentropy
paths -
- training path --- /data_set_name/training/train
- validation path --- /data_set_name/training/validation
- evaluation path --- /data_set_name/evalaution//
Algorithms
- for normal training value is training
GPU
- If utilizing the 2 gpus then hardcode the values of workers to 12 (each gpu carrys 6 workers) and choose the priority as low/medium/high.