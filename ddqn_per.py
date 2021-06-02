from replay.per.proportional import ProportionalPER
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from wrappers.video_recorder import VideoRecorder
from annealing.linear import LinearDecaySchedule
from trainers.tf.ddqn import DoubleDQNTrainer
from wrappers.frame_stack import FrameStack
from models.tf.nature_dqn import NatureDQN
from wrappers.logger import Logger
from rich.progress import *
from utils import pbar
from time import time

from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

import hydra
import gym

@hydra.main(config_name='config')
def main(cfg):
    env = gym.make(cfg.game + "NoFrameskip-v0")
    env = Logger(env)
    env = VideoRecorder(env, fps = 60)
    env = AtariPreprocessing(env)
    env = FrameStack(env)

    memory = ProportionalPER(cfg.memory_length)
    schedule = LinearDecaySchedule(mineps = cfg.mineps, maxeps = cfg.maxeps, length = cfg.decay_duration)

    trainer = DoubleDQNTrainer(
        optimizer = Adam(learning_rate = cfg.lr),
        criterion = MeanSquaredError(),
        num_actions = env.action_space.n,
        gamma = cfg.gamma
    )

    start = time()

    with pbar() as progress:
        task = progress.add_task("Training...", total = cfg.duration)

        steps = 0
        frames = []

        while steps < cfg.duration:
            state = env.reset()
            terminal = False

            while not terminal:
                action = trainer.act(state, schedule.epsilon, compute_value = True)
                next_state, reward, terminal, info = env.step(action)
                steps += 1
                progress.advance(task)
                if cfg.render:
                    env.render()
                schedule.update()

                error = (action - trainer.value) ** 2
                memory.append(state, action, next_state, reward, terminal, error)
                state = next_state

                if steps % cfg.checkpoint_freq == 0:
                    trainer.save(f'{steps}.h5')

                if steps % cfg.tau == 0:
                    trainer.update_target()

                if memory.exceeds(cfg.minibatch_size):
                    states, actions, next_states, rewards, terminals, indices, weights = memory.sample(cfg.minibatch_size)
                    loss = trainer.replay(states, actions, next_states, rewards, terminals, weights = weights)
                    progress.print(f"Episode: {env.episode}, Frame: {env.frame}, Highscore: {env.highscore}, Epsilon: {schedule.epsilon:.10f}, Loss: {loss:.10f}")
                    memory.update(indices, trainer.error)
        else:
            env.close()

if __name__ == '__main__':
    main()
