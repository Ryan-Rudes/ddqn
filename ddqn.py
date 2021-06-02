from gym.wrappers.atari_preprocessing import AtariPreprocessing
from wrappers.video_recorder import VideoRecorder
from annealing.linear import LinearDecaySchedule
from trainers.tf.ddqn import DoubleDQNTrainer
from replay.replay_memory import ReplayMemory
from wrappers.frame_stack import FrameStack
from wrappers.grayscale import Grayscale
from wrappers.resizing import Resizing
# from rich.progress import *
# from utils import pbar
from time import time

from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

import hydra
import gym

from environments.slitherio import Slitherio

@hydra.main(config_name='config')
def main(cfg):
    # env = gym.make(cfg.game + "NoFrameskip-v0")
    env = Slitherio("AI")
    env = VideoRecorder(env, fps = 60)
    env = Resizing(env, width = 84, height = 84)
    env = Grayscale(env)
    # env = AtariPreprocessing(env)
    env = FrameStack(env)
    env.start()

    memory = ReplayMemory(cfg.memory_length, (84, 84, 4))
    schedule = LinearDecaySchedule(mineps = cfg.mineps, maxeps = cfg.maxeps, length = cfg.decay_duration)

    trainer = DoubleDQNTrainer(
        optimizer = Adam(learning_rate = cfg.lr),
        criterion = MeanSquaredError(),
        num_actions = env.action_space.n,
        gamma = cfg.gamma
    )

    start = time()

    # with pbar() as progress:
    #     task = progress.add_task("Training...", total = cfg.duration)

    steps = 0
    frames = []

    while steps < cfg.duration:
        state = env.reset()
        terminal = False
        length = 0

        while not terminal:
            action = trainer.act(state, schedule.epsilon)
            next_state, reward, terminal, info = env.step(action)
            steps += 1
            length += 1
            # progress.advance(task)
            if cfg.render:
                env.render()
            schedule.update()
            memory.append(state, action, next_state, reward, terminal)
            state = next_state

            if steps % cfg.tau == 0:
                trainer.update_target()

            if steps % cfg.checkpoint_freq == 0:
                trainer.save(f'{steps}.h5')

            """
            if memory.exceeds(cfg.minibatch_size):
                minibatch = memory.sample(cfg.minibatch_size)
                loss = trainer.replay(*minibatch)
                progress.print(f"Loss: {loss:.10f}")
            """

        if memory.exceeds(cfg.minibatch_size):
            for i in range(length):
                minibatch = memory.sample(cfg.minibatch_size)
                loss = trainer.replay(*minibatch)
                # progress.print(f"Loss: {loss:.10f}")
                print('Loss:', loss)
    else:
        env.close()

if __name__ == '__main__':
    main()
