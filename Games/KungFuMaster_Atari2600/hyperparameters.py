class Hyperparameters(object):
    MAX_EPISODES = 100
    REPLY_START_SIZE = 4000
    UPDATE_FREQUENCY = 2
    LEARNING_RATE = 0.001
    REWARD_DECAY = 0.9
    E_GREEDY = 0.9
    TARGET_REPLACE_ITER = 1000
    MEMORY_SIZE = 2000
    BATCH_SIZE = 32
    E_GREEDY_INCRE = 0.00005

    WEIGHTS_SAVER_ITER = 2000
    OUTPUT_SAVER_ITER = 1000
    SAVED_NETWORK_PATH = './saved_network/'
    LOGS_DATA_PATH = './logs/'

    # Class Memory
    M_EPSILON = 0.01  # small amount to avoid zero priority
    M_ALPHA = 0.6  # [0~1] convert the importance of TD error to priority
    M_BETA = 0.4  # importance-sampling, from initial value increasing to 1
    M_BETA_INCRE = 0.001
    M_ABS_ERROR_UPPER = 1.  # clipped abs error
