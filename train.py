# Imports

# Hyperparameters
'''
shape of input layer (as long as sufficiently big??)
shape of output layer (must match input layer??)
shape of middle layers (as long as they connect)
learning rate
weights, biases of different layers

'''

# Weights and biases logging stuff
# We want to log accuracy of different agents' predictions over time
# We want to log agents' health bars
# Want to log system health bar over time

# Training

#XXX working here
def signal_handler(sig, frame):
    """
    Gracefully stop training and pickle the models
    :param sig:
    :param frame:
    :return:
    """
    print('You pressed Ctrl+C!')
    sys.exit(0)