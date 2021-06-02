from models.tf.nature_dqn import NatureDQN
from utils import epsilon_random

import tensorflow as tf
import numpy as np

class DQNTrainer:
    def __init__(self,
                 optimizer,
                 criterion,
                 num_actions,
                 gamma=0.99,
                 modelfn=None,
                 atari=True):
        if modelfn is None:
            self.model = NatureDQN(num_actions)
        else:
            self.model = modelfn()

        self.optimizer = optimizer
        self.criterion = criterion

        self.gamma = gamma
        self.num_actions = num_actions
        self.atari = atari

        self.updates = 0

    def __call__(self, input):
        return self.model(input)

    def act(self, observation, epsilon=1.0, compute_value=False):
        random_action = epsilon_random(epsilon)

        if not random_action or compute_value:
            if self.atari:
                observation = observation / 255.0

            q = self.model.predict(np.expand_dims(observation, axis = 0))[0]
            action = np.argmax(q)
            self.value = q[action]

        if random_action:
            return np.random.randint(self.num_actions)
        return action

    def predict(self, input):
        return self.model.predict(input)

    def replay(self, states, actions, next_states, rewards, terminals, **kwargs):
        if self.atari:
            states = states / 255.0
            next_states = next_states / 255.0

        pred_values = np.max(self.model(next_states), axis = 1)
        real_values = np.where(terminals, rewards, rewards + self.gamma * pred_values)
        with tf.GradientTape() as tape:
            selected_actions_one_hot = tf.one_hot(actions, self.num_actions)
            selected_action_values = tf.math.reduce_sum(self.model(states) * selected_actions_one_hot, axis = 1)
            loss = self.criterion(real_values, selected_action_values)
            if 'weights' in kwargs:
                loss = tf.math.reduce_mean(loss * kwargs['weights'])
                self.error = tf.math.abs(selected_action_values - real_values).numpy()
        vars = self.model.trainable_variables
        grads = tape.gradient(loss, vars)
        self.optimizer.apply_gradients(zip(grads, vars))
        self.updates += 1
        return loss.numpy()

    def load(self, filepath):
        self.model.load_weights(filepath)

    def save(self, filepath):
        self.model.save_weights(filepath)
