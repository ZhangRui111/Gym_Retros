import cv2
import retro
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Set log level: only output error.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Only use #0 GPU.
import time
import tensorflow as tf
import matplotlib as mlp
mlp.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from Hyper_parameters.hp_pri_dqn import Hyperparameters
from Utils.write_to_file import write_to_file_w
from Utils.write_to_file import write_to_file_a


def restore_parameters(sess, model):
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(hp.SAVED_NETWORK_PATH + model + '/')
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
        path_ = checkpoint.model_checkpoint_path
        step = int((path_.split('-'))[-1])
    else:
        # Re-train the network from zero.
        print("Could not find old network weights")
        step = 0
    return saver, step


def preprocessing(obser):
    x_t = cv2.cvtColor(cv2.resize(obser, (hp.IMAGE_LENGTH, hp.IMAGE_LENGTH)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    x_t = x_t/255  # [0, 1]
    x_t = np.expand_dims(x_t, 2)
    return x_t  # (96, 96)


def preprocessing_stack(obser):
    x_t = cv2.cvtColor(cv2.resize(obser, (hp.IMAGE_LENGTH, hp.IMAGE_LENGTH)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    x_t = x_t / 255  # [0, 1]
    obser_stack = np.stack(([x_t]*hp.AGENT_HISTORY_LENGTH), axis=2)
    return obser_stack  # (4, 96, 96)


def run_stargunner(env, RL, model, saver, load_step):
    total_steps = 0  # total steps after training begins.
    episodes = []  # episode's index.
    steps_episode = []  # steps for every single episode.
    rewards_episode = []  # rewards for every episodes.

    for i_episode in range(hp.MAX_EPISODES):
        print('episode:' + str(i_episode))
        episode_steps = 0
        obser_pool = []

        rewards = 0

        observation = env.reset()
        obser_pool.append(observation)
        obser_cv_stack = preprocessing_stack(observation)
        action = RL.choose_action(np.expand_dims(obser_cv_stack, axis=0))

        while True:
            env.render()
            if total_steps % hp.AGENT_HISTORY_LENGTH == 0:
                # RL choose action based on observation
                action = RL.choose_action(np.expand_dims(preprocessing_stack(observation), axis=0))
                # RL take action and get next observation and reward
            else:
                pass

            observation_, reward, done, info = env.step(action)
            rewards += reward
            obser_pool.append(observation_)

            if len(obser_pool) >= hp.AGENT_HISTORY_LENGTH+1:
                obser_cv_stack = preprocessing(obser_pool[0])  # (96, 96, 1)
                # concate four (96, 96, 1) to one (96, 96, 4)
                for i in range(1, hp.AGENT_HISTORY_LENGTH):
                    obser_cv = preprocessing(obser_pool[i])  # (96, 96, 1)
                    obser_cv_stack = np.concatenate((obser_cv_stack, obser_cv), axis=2)

                obser_cv_stack = obser_cv_stack.astype(np.float32)  # int8 --> float64
                obser_cv_stack_ = preprocessing_stack(obser_pool[hp.AGENT_HISTORY_LENGTH])
                obser_cv_stack_ = obser_cv_stack_.astype(np.float32)  # int8 --> float64

                # store this transition
                transition = np.append(obser_cv_stack, action)
                transition = np.append(transition, [reward])
                transition = np.append(transition, obser_cv_stack_)
                RL.memory.append(transition)
                if len(RL.memory) > hp.REPLY_MEMORY_SIZE:
                    RL.memory.popleft()
                # del all frames except for the newest frame.
                del obser_pool[:hp.AGENT_HISTORY_LENGTH]

            observation = observation_
            episode_steps += 1
            total_steps += 1

            if total_steps > hp.REPLY_START_SIZE:
                RL.learn()
                # store network's weights
                if total_steps % hp.WEIGHTS_SAVER_ITER == 0:
                    saver.save(RL.sess, hp.SAVED_NETWORK_PATH + model + '/' + '-' + model + '-' +
                               str(total_steps + load_step))
                    print('-----save weights-----')
                # save rewards
                if total_steps % hp.OUTPUT_SAVER_ITER == 0:
                    filename1 = hp.LOGS_DATA_PATH + model + '/rewards_total.txt'
                    write_to_file_a(filename1, str(np.vstack((episodes, rewards_episode))))
                    print('-----save outputs-----')

            if done:
                print('episode ' + str(i_episode) + ' | ' + str(rewards))
                steps_episode.append(episode_steps)
                rewards_episode.append(rewards)
                episodes.append(i_episode)
                break

    return np.vstack((episodes, rewards_episode)), np.vstack((episodes, steps_episode))


def run_stargunner_pri(env, RL, model, saver, load_step):
    total_steps = 0  # total steps after training begins.
    episodes = []  # episode's index.
    steps_episode = []  # steps for every single episode.
    rewards_episode = []  # rewards for every episodes.

    for i_episode in range(hp.MAX_EPISODES):
        print('episode:' + str(i_episode))
        episode_steps = 0
        obser_pool = []

        rewards = 0

        observation = env.reset()
        obser_pool.append(observation)
        obser_cv_stack = preprocessing_stack(observation)
        action = RL.choose_action(np.expand_dims(obser_cv_stack, axis=0))

        while True:
            env.render()
            if total_steps % hp.AGENT_HISTORY_LENGTH == 0:
                # RL choose action based on observation
                action = RL.choose_action(np.expand_dims(preprocessing_stack(observation), axis=0))
                # RL take action and get next observation and reward
            else:
                pass

            observation_, reward, done, info = env.step(action)
            rewards += reward
            obser_pool.append(observation_)

            if len(obser_pool) >= hp.AGENT_HISTORY_LENGTH+1:
                obser_cv_stack = preprocessing(obser_pool[0])  # (96, 96, 1)
                # concate four (96, 96, 1) to one (96, 96, 4)
                for i in range(1, hp.AGENT_HISTORY_LENGTH):
                    obser_cv = preprocessing(obser_pool[i])  # (96, 96, 1)
                    obser_cv_stack = np.concatenate((obser_cv_stack, obser_cv), axis=2)

                obser_cv_stack = obser_cv_stack.astype(np.float32)  # int8 --> float64
                obser_cv_stack_ = preprocessing_stack(obser_pool[hp.AGENT_HISTORY_LENGTH])
                obser_cv_stack_ = obser_cv_stack_.astype(np.float32)  # int8 --> float64

                # store this transition
                transition = np.append(obser_cv_stack, action)
                transition = np.append(transition, [reward])
                transition = np.append(transition, obser_cv_stack_)

                RL.store_transition(transition)
                # del all frames except for the newest frame.
                del obser_pool[:hp.AGENT_HISTORY_LENGTH]

            observation = observation_
            episode_steps += 1
            total_steps += 1

            if total_steps > hp.REPLY_START_SIZE:
                RL.learn()
                # store network's weights
                if total_steps % hp.WEIGHTS_SAVER_ITER == 0:
                    saver.save(RL.sess, hp.SAVED_NETWORK_PATH + model + '/' + '-' + model + '-' +
                               str(total_steps + load_step))
                    print('-----save weights-----')
                # save rewards
                if total_steps % hp.OUTPUT_SAVER_ITER == 0:
                    filename1 = hp.LOGS_DATA_PATH + model + '/rewards_total.txt'
                    write_to_file_a(filename1, str(np.vstack((episodes, rewards_episode))))
                    print('-----save outputs-----')

            if done:
                print('episode ' + str(i_episode) + ' | ' + str(rewards))
                steps_episode.append(episode_steps)
                rewards_episode.append(rewards)
                episodes.append(i_episode)
                break

    return np.vstack((episodes, rewards_episode)), np.vstack((episodes, steps_episode))


def main(model):
    env = retro.make(game='StarGunner-Atari2600')
    # env.unwrapped can give us more information.
    env = env.unwrapped
    rand_list = []

    if model == 'dqn_2015':
        from Brain.dqn_2015 import DeepQNetwork
        from Games.StarGunner_Atari2600.network_dqn_2015 import build_network
        rand = np.random.randint(1000)
        while rand in rand_list:
            rand = np.random.randint(1000)
        rand_list.append(rand)
        built_net = build_network(rand)
        # get the DeepQNetwork Agent
        RL = DeepQNetwork(built_net)
    elif model == "dqn_2013":
        from Brain.dqn_2013 import DeepQNetwork
        from Games.StarGunner_Atari2600.network_dqn_2013 import build_network
        rand = np.random.randint(1000)
        while rand in rand_list:
            rand = np.random.randint(1000)
        rand_list.append(rand)
        built_net = build_network(rand)
        # get the DeepQNetwork Agent
        RL = DeepQNetwork(built_net)
    elif model == "dueling_dqn":
        from Brain.dueling_dqn import DeepQNetwork
        from Games.StarGunner_Atari2600.network_dueling_dqn import build_network
        rand = np.random.randint(1000)
        while rand in rand_list:
            rand = np.random.randint(1000)
        rand_list.append(rand)
        built_net = build_network(rand)
        # get the DeepQNetwork Agent
        RL = DeepQNetwork(built_net)
    elif model == "pri_dqn":
        from Brain.pri_dqn import DeepQNetwork
        from Games.StarGunner_Atari2600.network_pri_dqn import build_network
        rand = np.random.randint(1000)
        while rand in rand_list:
            rand = np.random.randint(1000)
        rand_list.append(rand)
        built_net = build_network(rand)
        # get the DeepQNetwork Agent
        RL = DeepQNetwork(built_net)
    else:
        print("Warning: invalid code for algorithm! Running dqn_2015 instead!")
        from Brain.dqn_2015 import DeepQNetwork
        from Games.StarGunner_Atari2600.network_dqn_2015 import build_network
        rand = np.random.randint(1000)
        while rand in rand_list:
            rand = np.random.randint(1000)
        rand_list.append(rand)
        built_net = build_network(rand)
        # get the DeepQNetwork Agent
        RL = DeepQNetwork(built_net)

    saver, load_step = restore_parameters(RL.sess, model)

    # Calculate running time
    start_time = time.time()
    if model == "pri_dqn":
        results = run_stargunner_pri(env, RL, model, saver, load_step)
    else:
        results = run_stargunner(env, RL, model, saver, load_step)

    end_time = time.time()
    running_time = (end_time - start_time) / 60

    filename = hp.LOGS_DATA_PATH + model + "/running_time.txt"
    write_to_file_w(filename, str(running_time))

    return results


if __name__ == '__main__':
    hp = Hyperparameters()
    # # change different models here:
    # pri_dqn, double_dqn...
    result1 = main(model='pri_dqn')
