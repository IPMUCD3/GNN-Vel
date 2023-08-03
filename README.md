This code is aimed at creating a GNN model to derive the velocity of a galactic cluster, along the z axis. The role of the different scripts is listed bellow :


create dataset.py : Import the data from the Magneticum boxes and creates the graphs in the training, validation and testing sets.
constant.py : Define the hyperparameters (lmax, nlayers, etc.) of the models and some important functions.
networks.py : Define the GNN architecture.
training.py : Define the training routine functions, the bias correction.
main.py : The main code, it imports the dataset created with create dataset.py, create a GNN model with networks.py, and train it using the functions in training.py.


