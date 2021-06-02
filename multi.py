from gym.wrappers.atari_preprocessing import AtariPreprocessing
from wrappers.video_recorder import VideoRecorder
from annealing.linear import LinearDecaySchedule
from trainers.tf.ddqn import DoubleDQNTrainer
from replay.replay_memory import ReplayMemory
from wrappers.frame_stack import FrameStack
from utils import epsilon_random, pbar
from rich.progress import *
from random import choice
from time import time

from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

import numpy as np
import hydra
import gym

# MontezumaRevenge

games = ['Alien', 'Amidar', 'Assault', 'Asterix', 'Asteroids', 'Atlantis', 'BankHeist', 'BattleZone', 'BeamRider',
         'Berzerk', 'Bowling', 'Boxing', 'Breakout', 'Centipede', 'ChopperCommand', 'CrazyClimber', 'Defender',
         'DemonAttack', 'DoubleDunk', 'Enduro', 'FishingDerby', 'Freeway', 'Frostbite', 'Gopher', 'Gravitar',
         'Hero', 'IceHockey', 'Jamesbond', 'Kangaroo', 'Krull', 'KungFuMaster', 'MsPacman', 'NameThisGame',
         'Phoenix', 'Pitfall', 'Pong', 'PrivateEye', 'Qbert', 'Riverraid', 'RoadRunner', 'Robotank', 'Seaquest',
         'Skiing', 'Solaris', 'SpaceInvaders', 'StarGunner', 'Tennis', 'TimePilot', 'Tutankham', 'UpNDown',
         'Venture', 'VideoPinball', 'WizardOfWor', 'YarsRevenge', 'Zaxxon']

@hydra.main(config_name='config')
def main(cfg):
    memory = ReplayMemory(cfg.memory_length, (84, 84, 4))
    schedule = LinearDecaySchedule(mineps = cfg.mineps, maxeps = cfg.maxeps, length = cfg.decay_duration)

    trainer = DoubleDQNTrainer(
        optimizer = Adam(learning_rate = cfg.lr),
        criterion = MeanSquaredError(),
        num_actions = 16,
        gamma = cfg.gamma
    )

    start = time()

    with pbar() as progress:
        task = progress.add_task("Training...", total = cfg.duration)

        steps = 0
        frames = []

        while steps < cfg.duration:
            env = gym.make(choice(games) + "NoFrameskip-v0")
            env = VideoRecorder(env, fps = 60)
            env = AtariPreprocessing(env)
            env = FrameStack(env)

            state = env.reset()
            terminal = False

            while not terminal:
                if epsilon_random(schedule.epsilon):
                    action = env.action_space.sample()
                else:
                    q = trainer.model.predict(np.expand_dims(state / 255.0, axis = 0))[0][:env.action_space.n]
                    action = np.argmax(q)

                next_state, reward, terminal, info = env.step(action)
                reward = max(min(reward, 1), -1)
                steps += 1
                progress.advance(task)
                if cfg.render:
                    env.render()
                schedule.update()
                memory.append(state, action, next_state, reward, terminal)
                state = next_state

                if steps % cfg.tau == 0:
                    trainer.update_target()

                if steps % cfg.checkpoint_freq == 0:
                    trainer.save(f'{steps}.h5')

                if memory.exceeds(cfg.minibatch_size):
                    minibatch = memory.sample(cfg.minibatch_size)
                    loss = trainer.replay(*minibatch)
                    progress.print(f"Loss: {loss:.10f}")

            env.close()
        else:
            env.close()

if __name__ == '__main__':
    main()
