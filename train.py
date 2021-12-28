# Imports
import numpy as np
import torch
import signal
import sys

# Hyperparameters
'''
shape of input layer (as long as sufficiently big??)
shape of output layer (must match input layer??)
shape of middle layers (as long as they connect)
learning rate
weights, biases of different layers

'''
ITER_NO_BACKPROP = 3
DEVICE = 'cuda??'
GAMMA = 0.99


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Weights and biases logging stuff
# We want to log accuracy of different agents' predictions over time
# We want to log agents' health bars
# Want to log system health bar over time

# Training

# XXX working here to gracefully stop training
def signal_handler(sig, frame):
    """
    Gracefully stop training and pickle the models
    :param sig:
    :param frame:
    :return:
    """
    print('You pressed Ctrl+C!')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def train():
    # Init gradient optimizers
    # Initialize environment, state, reward, "episode done"
    # Init past state for actor2???
    done = False
    while not done:
        print("training")
        actions = []
        rewards = []
        values = []
        losses = []
        for i in range(ITER_NO_BACKPROP):
            pass
            # Get actor1 action
            # get actor2 prediction of next state
            # step in environment given action to get reward, s'
            # Update environment state, decrement agent hunger + attractiveness
            # add reward to rewards
            # calc actor2 loss, add to losses
            # get critic value, add to values
        # zero gradients
        # backprop actor2 based on diff of predicted s' and s'; losses
        # calc advantages from values and rewards
        # Advantage: what specific action is worth compared to any average action in that state
        # A = Q(s, a) - V(s)
        # Bellman Optimality: Q(s, a) = r_t+1 + gamma * V(s')
        # A = r_t+1(s) + gamma * V(s') - V(s)
        # backprop actor1 from advantages
        # backprop critic from diff of reward and predicted values


def compute_q_vals(next_critic_value, rewards, values):
    qvals = np.zeros_like(values)
    for t in reversed(range(len(rewards))):
        qval = rewards[t] + GAMMA * next_critic_value
        qvals[t] = qval
    return qvals
