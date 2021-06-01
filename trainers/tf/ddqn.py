from models.tf.nature_dqn import NatureDQN

import tensorflow as tf
import numpy as np

class DoubleDQNTrainer:
    def __init__(self,
                 optimizer,
                 criterion,
                 num_actions,
                 gamma=0.99):
        self.model = NatureDQN(num_actions)
        self.target = NatureDQN(num_actions)

        self.optimizer = optimizer
        self.criterion = criterion

        self.gamma = gamma
        self.num_actions = num_actions

        self.updates = 0

    def __call__(self, input):
        return self.model(input)

    def act(self, observation, epsilon=1.0):
        if np.random.random() < epsilon:
            return np.random.randint(self.num_actions)
        q = self.model.predict(np.expand_dims(observation / 255.0, axis = 0))[0]
        return np.argmax(q)

    def predict(self, input):
        return self.model.predict(input)

    def replay(self, states, actions, next_states, rewards, terminals):
        pred_values = np.max(self.target(next_states), axis = 1)
        real_values = np.where(terminals, rewards, rewards + self.gamma * pred_values)
        with tf.GradientTape() as tape:
            selected_actions_one_hot = tf.one_hot(actions, self.num_actions)
            selected_action_values = tf.math.reduce_sum(self.model(states) * selected_actions_one_hot, axis = 1)
            loss = self.criterion(real_values, selected_action_values)
        vars = self.model.trainable_variables
        grads = tape.gradient(loss, vars)
        self.optimizer.apply_gradients(zip(grads, vars))
        self.updates += 1
        return loss.numpy()

    def load(self, filepath, target=False):
        if target:
            self.target.load_weights(filepath)
        else:
            self.model.load_weights(filepath)

    def save(self, filepath, target=False):
        if target:
            self.target.save_weights(filepath)
        else:
            self.model.save_weights(filepath)

    def update_target(self):
        self.target.set_weights(self.model.get_weights())
