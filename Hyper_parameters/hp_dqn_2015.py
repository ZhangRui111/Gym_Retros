"""
This is all hyper-parameters needed in DQN-2015 Algorithm,
you can modify them here.
"""


class Hyperparameters(object):
    def __init__(self):
        self.IMAGE_LENGTH = 96
        self.N_FEATURES = 128
        self.N_ACTIONS = 8

        self.MINIBATCH_SIZE = 32
        self.REPLY_MEMORY_SIZE = 1000000
        self.AGENT_HISTORY_LENGTH = 4
        self.TARGET_NETWORK_UPDATE_FREQUENCY = 10000
        self.DISCOUNT_FACTOR = 0.99
        self.LEARNING_RATE = 0.00025
        self.INITIAL_EXPLOR = 0
        self.FINAL_EXPLOR = 0.9
        self.FINAL_EXPLOR_FRAME = 1000000
        self.REPLY_START_SIZE = 500  # 50000

        self.MAX_EPISODES = 100000
        self.WEIGHTS_SAVER_ITER = 20000
        self.OUTPUT_SAVER_ITER = 10000
        self.OUTPUT_GRAPH = True
        self.SAVED_NETWORK_PATH = './saved_network/'
        self.LOGS_DATA_PATH = './logs/'
