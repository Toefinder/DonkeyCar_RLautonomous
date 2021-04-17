"""
file: ddqn.py
author: Felix Yu
date: 2018-09-12
original: https://github.com/flyyufelix/donkey_rl/blob/master/donkey_rl/src/ddqn.py
"""
import argparse
import os
import random
import signal
import sys
import uuid
from collections import deque

import cv2
import gym
import gym_donkeycar
import numpy as np
import tensorflow as tf
# from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt # debug
# logging
# from stable_baselines import logger
# from stable_baselines.common import explained_variance, tf_util, TensorboardWriter
# from stable_baselines.common.tf_util import mse, total_episode_reward_logger
import math
import datetime
from imageprocess import detect_edge
img_rows, img_cols = 80, 80
# Convert image into Black and white
img_channels = 4  # We stack 4 frames
steering_binsize = 5
throttle_binsize = 5
steering_min = -0.5
steering_max = 0.5
throttle_min = 0
throttle_max = 1

env_steering_min = -1
env_steering_max = 1
env_throttle_min = 0
env_throttle_max = 1


class DQNAgent:
    def __init__(self, state_size, action_space, train=True):
        self.t = 0
        self.max_Q = 0
        self.train = train

        # Get size of state and action
        self.state_size = state_size
        self.action_space = action_space
        self.action_size = action_space

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 1e-4
        if self.train:
            self.epsilon = 1.0
            self.initial_epsilon = 1.0
        else:
            self.epsilon = 1e-6
            self.initial_epsilon = 1e-6
        self.epsilon_min = 0.02
        self.batch_size = 64
        self.train_start = 100
        self.explore = 10000
        
        # Create replay memory using deque
        self.memory = deque(maxlen=10000)

        # Create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # Copy the model to target model
        # --> initialize the target model so that the parameters of model & target model to be same
        self.update_target_model()

    def build_model(self):
        model = Sequential()
        model.add(
            Conv2D(24, (5, 5), strides=(2, 2), padding="same", input_shape=(img_rows, img_cols, img_channels))
        )  # 80*80*4
        model.add(Activation("relu"))
        model.add(Conv2D(32, (5, 5), strides=(2, 2), padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(64, (5, 5), strides=(2, 2), padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same"))
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))

        # 25 categorical bins for Steering angles (5) and Throttle (5)
        model.add(Dense(25, activation="linear", name="steering_throttle"))

        adam = Adam(lr=self.learning_rate)
        model.compile(loss="mse", optimizer=adam)

        return model

    def rgb2gray(self, rgb):
        """
        take a numpy rgb image return a new single channel image converted to greyscale
        """
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def process_image(self, obs):
        global LANE_SEGMENTATION 
        obs = self.rgb2gray(obs)
        obs = cv2.resize(obs, (img_rows, img_cols))
#         print("in process_image") # debug 
#         plt.imshow(obs) # debug
        if LANE_SEGMENTATION: obs = detect_edge(obs)
        return obs

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # Get action from model using epsilon-greedy policy
    def get_action(self, s_t):
        if np.random.rand() <= self.epsilon:
            sampled_steering = np.interp(self.action_space.sample()[0], [env_steering_min, env_steering_max], [steering_min, steering_max])
            sampled_throttle = np.interp(self.action_space.sample()[1], [env_throttle_min, env_throttle_max], [throttle_min, throttle_max])
            steering = linear_bin(sampled_steering, size=steering_binsize, min_val=steering_min, max_val=steering_max)
            throttle = linear_bin(sampled_throttle, size=throttle_binsize, min_val=throttle_min, max_val=throttle_max)
            return throttle_binsize * np.argmax(steering) + np.argmax(throttle)
        else:
            # print("Return Max Q Prediction")
            q_value = self.model.predict(s_t)
            
            # Convert q array to steering and throttle value
            return np.argmax(q_value[0])

    def replay_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= (self.initial_epsilon - self.epsilon_min) / self.explore

    def train_replay(self):
        if len(self.memory) < self.train_start:
            return

        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)

        state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch)
        
        state_t = np.concatenate(state_t)
#         print(state_t.shape) # debug
        state_t1 = np.concatenate(state_t1)
        targets = self.model.predict(state_t)
        self.max_Q = np.max(targets[0])
        target_val = self.model.predict(state_t1)
        target_val_ = self.target_model.predict(state_t1)
        for i in range(batch_size):
            if terminal[i]:
                targets[i][action_t[i]] = reward_t[i]
            else:
                a = np.argmax(target_val[i])
                targets[i][action_t[i]] = reward_t[i] + self.discount_factor * (target_val_[i][a])

        self.model.train_on_batch(state_t, targets)

    def load_model(self, name):
        self.model.load_weights(name)

    # Save the model which is under training
    def save_model(self, name):
        self.model.save_weights(name)

    def save_ready_model(self, name):
        self.model.save(name)


# Utils Functions #


def linear_bin(a, size=15, min_val=-1, max_val=1):
    """
    Convert a value to a categorical array.

    Parameters
    ----------
    a : int or float
        A value between min_val and max_val

    Returns
    -------
    list of int
        A list of length "size" with one item set to 1, which represents the linear value, and all other items set to 0.
    """
    a = a - min_val
    b = round(a / ((max_val-min_val) / (size-1)))
    arr = np.zeros(size)
    arr[int(b)] = 1
    return arr


def linear_unbin(arr, size=15, min_val=-1, max_val=1):
    """
    Convert a categorical array to value.

    See Also
    --------
    linear_bin
    """
    if not len(arr) == size:
        raise ValueError("Illegal array length, must be {}".format(size))
    b = np.argmax(arr)
    a = b * ((max_val-min_val) / (size-1)) + min_val
    return a

def ep_over_fn(self):
    # we have a few initial frames on start that are sometimes very large CTE when it's behind
    # the path just slightly. We ignore those.
    global DEBUG_MODE
    global episode_len
    global MAX_EPISODE_LEN
    if math.fabs(self.cte) > 2 * self.max_cte:
        pass
    elif self.cte > 1/2 * self.max_cte:
        if DEBUG_MODE: print(f"game over stray too much right: cte {self.cte}")
        self.over = True
    elif self.cte < (-1) * self.max_cte:
        if DEBUG_MODE: print(f"game over stray too much left: cte {self.cte}")
        self.over = True
    elif self.hit != "none":
        if DEBUG_MODE: print(f"game over: hit {self.hit}")
        self.over = True
    elif episode_len >= MAX_EPISODE_LEN:
        if DEBUG_MODE: print(f"game won: survived {MAX_EPISODE_LEN} timesteps in episode")
        self.over = True
    elif self.missed_checkpoint:
        if DEBUG_MODE: print("missed checkpoint")
        self.over = True
    elif self.dq:
        if DEBUG_MODE: print("disqualified")
        self.over = True

def calc_reward(self, done):
    # global MAX_EPISODE_LEN
    # global episode_len
    # if done:
        # return -5.0

    # if done and episode_len >= MAX_EPISODE_LEN:
    #     return 100 # high reward for surviving the entire time

    if done:
        if self.cte > 1/2 * self.max_cte or self.cte < (-1) * self.max_cte:
            return -5.0
        
        elif self.hit != "none":
            return -5.0

        # else:
        #     return 0


    # going fast close to the center of lane yields best reward
    return 1.0 - (math.fabs(self.cte)/self.max_cte)
        
def run_ddqn(args):
    """
    run a DDQN training session, or test it's result, with the donkey simulator
    """

#     # only needed if TF==1.13.1
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     sess = tf.Session(config=config)
#     K.set_session(sess)
    
    # only use one GPU
    gpu_id = args.gpu
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpu_devices[gpu_id], 'GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[gpu_id], True)
    
#     # Check that GPU is used
#     tf.debugging.set_log_device_placement(True)
#     # Place tensors on the CPU
#     a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
#     b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
#     # Run on the GPU
#     c = tf.matmul(a, b)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/reward/train/' + current_time 
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    
    # number of episodes
    EPISODES = args.eps 
    # debug mode
    global DEBUG_MODE
    DEBUG_MODE = args.debug_mode
#     print(DEBUG_MODE) # debug

    # maximum episode length to decide when to stop the episode
    global MAX_EPISODE_LEN
    MAX_EPISODE_LEN = args.max_ep_len
    global episode_len # tracks the length of the episodes
    episode_len = 0
#     print(MAX_EPISODE_LEN) # debug
    global LANE_SEGMENTATION
    LANE_SEGMENTATION = args.lane_segment

    conf = {
        "exe_path": args.sim,
        "host": "127.0.0.1",
        "port": args.port,
        "body_style": "donkey",
        "body_rgb": (128, 128, 128),
        "car_name": "me",
        "font_size": 100,
        "racer_name": "DDQN",
        "country": "SG",
        "bio": "Learning to drive w DDQN RL",
        "guid": str(uuid.uuid4()),
        "max_cte": 10,
    }

    # Construct gym environment. Starts the simulator if path is given.
    env = gym.make(args.env_name, conf=conf)
    
    # Set custom reward function and episode over function
    env.set_episode_over_fn(ep_over_fn)
    env.set_reward_fn(calc_reward)
    
    # not working on windows...
    def signal_handler(signal, frame):
        print("catching ctrl+c")
        env.unwrapped.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGABRT, signal_handler)

    # Get size of state and action from environment
    state_size = (img_rows, img_cols, img_channels)
    action_space = env.action_space  # Steering and Throttle

    try:
        agent = DQNAgent(state_size, action_space, train=not args.test)

        # throttle = args.throttle  # Set throttle as constant value

        episodes = []

        if os.path.exists(args.model):
            print("load the saved model")
            agent.load_model(args.model)

        for e in range(EPISODES):

            print("Episode: ", e)

            done = False

            episode_len = 0
            episode_total_reward = 0
#             episode_all_speed = np.array([])
#             episode_all_cte = np.array([])
            episode_average_speed = 0
            episode_average_cte = 0
            episode_average_throttle = 0
            episode_average_steering = 0

            obs = env.reset()
            
            x_t = agent.process_image(obs)

            s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
            # In Keras, need to reshape
            s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  # 1*80*80*4
            
            
            while not done:

                # Get action for the current state and go one step in environment
                
                steering_throttle = agent.get_action(s_t)
                # print("action = {}".format(steering_throttle)) # debug steering_throttle

                steering_array = np.zeros(steering_binsize)
                steering_array[steering_throttle // throttle_binsize] = 1 
                steering = linear_unbin(steering_array, size=steering_binsize, min_val=steering_min, max_val=steering_max)
                # print("steering = {}".format(steering))# debug steering_throttle
                throttle_array = np.zeros(throttle_binsize)
                throttle_array[steering_throttle % throttle_binsize] = 1 
                throttle = linear_unbin(throttle_array, size=throttle_binsize, min_val=throttle_min, max_val=throttle_max)
                # print("throttle = {}".format(throttle)) # debug steering_throttle
                
                action = [steering, throttle]
                next_obs, reward, done, info = env.step(action)
                
                if DEBUG_MODE and agent.t % 10 == 0: 
                    print("CTE:{}".format(info["cte"])) # debug
#                 print("Speed:{}".format(info["speed"])) # debug

#                 if math.fabs(info["cte"]) > 100 and episode_len > 100:
#                     print("Stray too far from the center") # debug
#                     done = True
                episode_total_reward += reward
#                 episode_all_speed = np.append(episode_all_speed, np.array([info["speed"]]))
#                 episode_all_cte = np.append(episode_all_cte, np.array([info["cte"]]))
                episode_average_speed = (episode_average_speed * episode_len + info["speed"])/ (episode_len +1) 
                episode_average_cte = (episode_average_cte * episode_len + info["cte"])/ (episode_len +1) 
#                lap = info["lap"] # debug lap info
                episode_average_throttle = (episode_average_throttle * episode_len + throttle)/ (episode_len +1) 
                episode_average_steering = (episode_average_steering * episode_len + steering)/ (episode_len +1) 

                x_t1 = agent.process_image(next_obs)
                # plt.imshow(agent.process_image(next_obs)) # debug
                # plt.savefig("{}.png".format(agent.t)) # debug

                x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)  # 1x80x80x1
                s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)  # 1x80x80x4

                # Save the sample <s, a, r, s'> to the replay memory
                agent.replay_memory(s_t, steering_throttle, reward, s_t1, done)
                agent.update_epsilon()

                if agent.train:
                    agent.train_replay()

                s_t = s_t1
                agent.t = agent.t + 1
                episode_len = episode_len + 1
                if agent.t % 30 == 0:
                    print(
                        "EPISODE",
                        e,
                        "TIMESTEP",
                        agent.t,
                        "/ ACTION",
                        action,
                        "/ REWARD",
                        reward,
                        "/ EPISODE LENGTH",
                        episode_len,
                        "/ Q_MAX ",
                        agent.max_Q,
			            # "/ LAP ",
			            # lap
                    )


                if done:

                    # Every episode update the target model to be same with model
                    agent.update_target_model()

                    episodes.append(e)

                    # Save model for each episode
                    if agent.train:
                        agent.save_model(args.model)

                    print(
                        "episode:",
                        e,
                        "  memory length:",
                        len(agent.memory),
                        "  epsilon:",
                        agent.epsilon,
                        " episode length:",
                        episode_len,
                        " episode total reward:", 
                        episode_total_reward
                    )
                    
                    # Use writer to save the result
                    with train_summary_writer.as_default():
                        tf.summary.scalar('episode length', episode_len, step=e)
                        tf.summary.scalar('episode total reward', episode_total_reward, step=e)
                        tf.summary.scalar('episode average reward', episode_total_reward/episode_len, step=e)
#                         tf.summary.scalar('episode average speed verify', np.sum(episode_all_speed)/episode_len, step=e)
#                         tf.summary.scalar('episode average cte verify', np.sum(episode_all_cte)/episode_len, step=e)
                        tf.summary.scalar('episode average speed', episode_average_speed, step=e)
                        tf.summary.scalar('episode average cte', episode_average_cte, step=e)
                        tf.summary.scalar('episode average throttle', episode_average_throttle, step=e)
                        tf.summary.scalar('episode average steering', episode_average_steering, step=e)
        if agent.train:
            agent.save_ready_model(args.model[:-3]+"_ready.h5")
    except KeyboardInterrupt:
        print("stopping run...")
    finally:
        env.unwrapped.close()


if __name__ == "__main__":

    # Initialize the donkey environment
    # where env_name one of:
    env_list = [
        "donkey-warehouse-v0",
        "donkey-generated-roads-v0",
        "donkey-avc-sparkfun-v0",
        "donkey-generated-track-v0",
        "donkey-roboracingleague-track-v0",
        "donkey-waveshare-v0",
        "donkey-minimonaco-track-v0",
        "donkey-warren-track-v0",
    ]

    parser = argparse.ArgumentParser(description="ddqn")
    parser.add_argument(
        "--sim",
        type=str,
        default="manual",
        help="path to unity simulator. maybe be left at manual if you would like to start the sim on your own.",
    )
    parser.add_argument("--model", type=str, default="model/rl_driver.h5", help="path to model")
    parser.add_argument("--test", action="store_true", help="agent uses learned model to navigate env")
    parser.add_argument("--port", type=int, default=9091, help="port to use for websockets")
    parser.add_argument("--throttle", type=float, default=0.3, help="constant throttle for driving")
    parser.add_argument(
        "--env_name", type=str, default="donkey-generated-track-v0", help="name of donkey sim environment", choices=env_list
    )
    parser.add_argument("--gpu", type=int, default=0, help="name of GPU to use")
    parser.add_argument("--debug_mode", type=int, default=0, help="debug mode")
    parser.add_argument("--eps", type=int, default=10000, help="number of episodes to train for")
    parser.add_argument("--max_ep_len", type=int, default=2000, help="maximum length per episode") 
    parser.add_argument("--lane_segment", type=int, default=0, help="whether to perform lane segmentation") 


    args = parser.parse_args()
    

    run_ddqn(args)

    

    
