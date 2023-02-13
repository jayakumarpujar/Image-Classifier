@jayakumar

''' This is to run the specified algorithm'''

from cgi import test
import os
import yaml



with open('hyperparameter.yaml') as file:
    hyperparameters = yaml.load(file, Loader = yaml.FullLoader)

algorithm = hyperparameters['algorithm']

if __name__ == "__main__":

    if str(algorithm) == 'training':
        print("Running training")
        os.system('python3 training.py')

    elif str(algorithm) == 'training':

        from training import TrainAndTest 

        model_training = TrainAndTest()
        model_training.train()
        model_training.test()
        model_training.model_register()
