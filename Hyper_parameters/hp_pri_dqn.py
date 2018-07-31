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
        self.REPLY_START_SIZE = 50000  # 50000

        self.MAX_EPISODES = 100000
        self.WEIGHTS_SAVER_ITER = 20000
        self.OUTPUT_SAVER_ITER = 10000
        self.OUTPUT_GRAPH = False
        self.SAVED_NETWORK_PATH = './saved_network/'
        self.LOGS_DATA_PATH = './logs/'

        # Class Memory
        self.M_EPSILON = 0.01  # small amount to avoid zero priority
        self.M_ALPHA = 0.6  # [0~1] convert the importance of TD error to priority
        self.M_BETA = 0.4  # importance-sampling, from initial value increasing to 1
        self.M_BETA_INCRE = 0.001
        self.M_ABS_ERROR_UPPER = 1.  # clipped abs error
