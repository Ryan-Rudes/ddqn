from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium import webdriver

from gym.envs.classic_control.rendering import SimpleImageViewer
from selenium.common.exceptions import TimeoutException
from gym.envs.registration import EnvSpec
from io import BytesIO
from tqdm import tqdm
import numpy as np
import mss.tools
import random
import cv2
import gym
import mss

def try_forever(func, *args, **kwargs):
    while True:
        try:
            return func(*args, **kwargs)
        except:
            continue

class Slitherio(gym.Env):
    spec = EnvSpec("Slitherio-v0", nondeterministic = True)

    def __init__(self, nickname):
        self.nickname = nickname
        self.xpaths = {
            'nickname': '/html/body/div[2]/div[4]/div[1]/input',
            'mainpage': '/html/body/div[2]',
            'scorebar': '/html/body/div[13]/span[1]/span[2]'
        }
        self.monitor = {"top": 79, "left": 0, "width": 500, "height": 290}
        self.viewer = None
        self.browser = None
        self.last_observation = None

        self.observation_space = gym.spaces.Box(low = 0, high = 255, shape = (84, 84), dtype = np.uint8)
        self.action_space = gym.spaces.Discrete(32)

    def game_is_not_over(self):
        return self.browser.find_element_by_xpath(self.xpaths['mainpage']).value_of_css_property("display") == "none"

    def is_terminal(self):
        return self.browser.find_element_by_xpath(self.xpaths['mainpage']).value_of_css_property("display") != "none"

    def wait_until_can_enter_nickname(self):
        WebDriverWait(self.browser, 20).until(EC.element_to_be_clickable((By.XPATH, self.xpaths['nickname'])))

    def enter_nickame(self, nickname):
        self.wait_until_can_enter_nickname()
        field = self.browser.find_element_by_xpath(self.xpaths['nickname'])
        field.send_keys(self.nickname)
        return field

    def begin(self, field):
        self.wait_until_can_enter_nickname()
        field.send_keys(Keys.ENTER)

    def wait_until_game_has_loaded(self):
        WebDriverWait(self.browser, 1000).until(EC.invisibility_of_element((By.XPATH, self.xpaths['mainpage'])))

    def start(self):
        options = Options()
        options.add_experimental_option("useAutomationExtension", False)
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        self.browser = webdriver.Chrome(options = options)
        self.browser.set_window_size(500, 290)
        self.browser.set_window_position(0, 0)
        self.browser.set_page_load_timeout(30)
        self.restart()

    def restart(self):
        self.browser.get("http://slither.io")
        self.wait_until_can_enter_nickname()
        self.field = self.enter_nickame(self.nickname)

    def handle(e):
        if not isinstance(e, TimeoutException):
            print (str(e))
        self.restart()

    def reset(self):
        while True:
            try:
                self.begin(self.field)
                self.wait_until_game_has_loaded()
                self.score = try_forever(self.get_score)
                return self.observe()
            except Exception as e:
                self.handle(e)

    def observe(self):
        im = mss.mss().grab(self.monitor)
        im = np.array(im)[:, :, :3]
        # im = self.preprocess(im)
        self.last_observation = im
        return im

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = SimpleImageViewer()

        if self.last_observation is None:
            self.viewer.imshow(self.observe())
        else:
            self.viewer.imshow(self.last_observation)

        return None if mode == 'human' else self.last_observation

    def close(self):
        self.browser.close()
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def preprocess(self, frame):
        return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (84, 84)) # (145, 250))

    def sample(self):
        angle = np.random.random() * 2 * np.pi
        acceleration = int(np.random.random() > 0.5)
        return angle, acceleration

    def get_score(self):
        return int(self.browser.find_element_by_xpath(self.xpaths['scorebar']).text)

    def compute_reward(self):
        new_score = try_forever(self.get_score)
        reward = new_score - self.score
        self.score = new_score
        return reward

    """
    def step(self, angle, acceleration):
        angle *= 2 * np.pi
        x, y = np.cos(angle) * 360, np.sin(angle) * 360
        self.browser.execute_script("window.xm = %s; window.ym = %s; window.setAcceleration(%d);" % (x, y, acceleration))
        return self.observe(), self.compute_reward(), self.is_terminal(), {}
    """

    def step(self, action):
        try:
            if action >= 16:
                acceleration = 1
                action -= 16
            else:
                acceleration = 0

            angle = action / 8 * np.pi
            x, y = np.cos(angle) * 360, np.sin(angle) * 360
            self.browser.execute_script("window.xm = %s; window.ym = %s; window.setAcceleration(%d);" % (x, y, acceleration))
            return self.observe(), self.compute_reward(), self.is_terminal(), {}
        except Exception as e:
            observation = self.observe()
            self.handle(e)
            return observation, 0, True, {}
