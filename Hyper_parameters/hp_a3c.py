"""
This is all hyper-parameters needed in A3C Algorithm,
you can modify them here.
"""


class Hyperparameters(object):
    def __init__(self):
        self.IMAGE_LENGTH = 96
        self.AGENT_HISTORY_LENGTH = 4
        self.N_FEATURES = 4
        self.N_ACTIONS = 2

        self.GAME = 'CartPole-v0'
        self.N_WORKERS = 4  # or we can get the number of available CPUs by multiprocessing.cpu_count()
        self.GLOBAL_NET_SCOPE = 'Global_Net'
        self.UPDATE_GLOBAL_ITER = 10
        self.DISCOUNT_FACTOR = 0.9
        self.ENTROPY_BETA = 0.001
        self.LR_A = 0.001  # learning rate for actor
        self.LR_C = 0.001    # learning rate for critic

        self.MAX_GLOBAL_EPISODES = 1000
        self.WEIGHTS_SAVER_ITER = 20000
        self.OUTPUT_SAVER_ITER = 10000
        self.OUTPUT_GRAPH = False
        self.SAVED_NETWORK_PATH = './saved_network/'
        self.LOGS_DATA_PATH = './logs/'
