import gym
import time
import logging
import shutil
import os
import sys
# rdeg
import tensorflow as tf
# vxc
# import gcawdasdasw dwqad xcv
import gc
gc.enable()
# bb

# zcvbxvcb
from modules.dqn import *
from modules.env_utils import *


class ModelDQNN:
    def __init__(self, log_dir):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.InteractiveSession(config=config)
        self.saver = tf.train.Saver(tf.all_variables())
        self.log_dir = log_dir

    def init_dqnn(self, state_size, action_size):
        n = 0
        self.journalist = tf.summary.FileWriter(self.log_dir)
        self.agent = DQN(state_size=state_size,
                    action_size=action_size,
                    session=self.session,
                    summary_writer=self.journalist,
                    exploration_period=1000000,
                    minibatch_size=32,
                    discount_factor=0.99,
                    experience_replay_buffer=1000000,
                    target_qnet_update_frequency=20000,  # 30000 if UseDoubleDQN else 10000, ## Tuned DDQN
                    initial_exploration_epsilon=1.0,
                    final_exploration_epsilon=0.1,
                    reward_clipping=1.0,
                    DoubleDQN=True)

    def load_model(self, model_patn):
        self.saver.restore(self.session, model_patn)

    def save_model(self, model_path):
        self.saver.save(self.session, model_path)

    def train(self, batch_states, batch_newstates, batch_actions):
        predicted_actions = [self.agent.action(state, training=True) for state in batch_states]
        batch_rewards = [self.estimate_action(predicted_action, action) for predicted_action, action
                         in zip(predicted_actions, batch_actions)]
        self.agent.train(batch_states, batch_newstates, batch_rewards, batch_actions)

    def predict(self, state):
        return self.agent.action(state, training=False)

    def estimate_action(self, predicted_action, action):
        return 1 - abs(action - predicted_action)
