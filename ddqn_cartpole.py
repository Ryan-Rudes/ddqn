from annealing.exponential import ExponentialDecaySchedule
from models.tf.cartpole_dqn import CartPoleDQN
from replay.replay_memory import ReplayMemory
from trainers.tf.ddqn import DoubleDQNTrainer
from rich.progress import *
from utils import pbar
from time import time

from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

import hydra
import gym

@hydra.main(config_name='config_cartpole')
def main(cfg):
    env = gym.make("CartPole-v1")
    memory = ReplayMemory(cfg.memory_length, (4,), dtype = 'float32')
    schedule = ExponentialDecaySchedule(mineps = cfg.mineps, maxeps = cfg.maxeps, decay = cfg.exploration_decay)

    trainer = DoubleDQNTrainer(
        optimizer = Adam(learning_rate = cfg.lr),
        criterion = MeanSquaredError(),
        num_actions = env.action_space.n,
        gamma = cfg.gamma,
        modelfn = lambda: CartPoleDQN(),
        atari = False
    )

    start = time()

    with pbar() as progress:
        task = progress.add_task("Training...", total = cfg.duration)

        highscore = 0
        steps = 0
        frames = []
        episode = 1

        while steps < cfg.duration:
            state = env.reset()
            terminal = False
            score = 0

            while not terminal:
                action = trainer.act(state, schedule.epsilon)
                next_state, reward, terminal, info = env.step(action)
                score += reward
                highscore = max(highscore, score)
                steps += 1
                progress.advance(task)
                if cfg.render:
                    env.render()

                memory.append(state, action, next_state, reward, terminal)
                state = next_state

                if steps % cfg.checkpoint_freq == 0:
                    trainer.save(f'{steps}.h5')

                if memory.exceeds(cfg.minibatch_size):
                    minibatch = memory.sample(cfg.minibatch_size)
                    loss = trainer.replay(*minibatch)
                    schedule.update()
                    progress.print(f"Episode: {episode}, Epsilon: {schedule.epsilon:.10f} Highscore: {highscore}")

            trainer.update_target()
            episode += 1
        else:
            env.close()

if __name__ == '__main__':
    main()
