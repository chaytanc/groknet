import inspect
import os
import numpy as np
from PIL import Image
from selenium import webdriver
from selenium.webdriver import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from text_generator import TextGenerator
import torch
from observer import Observer


# Should actions be inherited by agents and then do agent.click etc...?
# Just use composition / each is passed in an instance of actions that it can enact (then
# future versions can alter what actions it does easily)

# HYPERPARAMETERS
hp = {'phrase_hidden_size' : 256, 'phrase_num_layers' : 6, 'phrase_seq_len' : 20}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actions(Observer):
    def __init__(self, state_size, headless=True, init_url="https://www.google.com"):
        self.headless = headless
        self.init_url = init_url
        self.driver = webdriver.Firefox()
        self.driver.get(init_url)
        self.wait = WebDriverWait(self.driver, timeout=3)
        fake_state = torch.zeros(state_size)
        # self.phrase_generator = TextGenerator(
        #     fake_state, hp['phrase_hidden_size'],
        #     hp['phrase_num_layers'], hp['phrase_seq_len']
        # )

    def click(self):
        """
        Click on the top link of google
        :return:
        """
        try:
            result_css = "//div[@id='search']//div[@class='g']/div[@class='rc']/div[@class='r']/a"
            self.wait.until(EC.element_to_be_clickable((By.XPATH, result_css)).click())
        except Exception as e:
            #XXX logging mechanism would be nice
            print("--- Failed to click", e)

    #XXX working to figure out how to generate phrase to google
    #XXX need to implement sequence length output from actor1 (could for now just use plan length and duplicate code to get
    # plan length later / make more general)
    def search(self, state):
        """
        Search the given phrase in google. Phrase to search is decided by agent.
        For simplicity right now could just decide to search most common word on page.
        :param phrase: the word or phrase to search in google
        :param sequence_len: the lambda of the geo distribution of length of the phrase to generate
        :return:
        """
        #XXX working here to generate phrase to search given the state
        phrase = self.phrase_generator.recurring_forward(state, "")
        # self.driver.get(phrase)
        self.driver.get(self.init_url)
        search_bar = self.driver.find_element(By.NAME, 'q')
        search_bar = self.wait.until(EC.element_to_be_clickable(search_bar))
        # search_bar = self.wait.until(EC.element_to_be_clickable((By.NAME, 'q')))
        # search_button = self.driver.find_element(By.NAME, 'btnK')
        search_bar.send_keys(phrase)
        search_bar.send_keys(Keys.ENTER)
        # search_button.click()

    def transfer_learn(self, architecture, other_architecture):
        """
        Restructure architecture by getting other network's prediction of best child architecture
        and output / spawn a new agent

        :return:
        """

    def end(self):
        self.driver.close()
        self.driver.quit()


def get_actions_num_actions(cls):
    actions = inspect.getmembers(cls, predicate=inspect.isfunction)
    return actions, len(actions)

# NOTE bad idea -- map to output the size of number of possible actions instead
# def map_actions_to_numbers(number, cls):
#     actions = inspect.getmembers(cls, predicate=inspect.isfunction)
#     num_actions = len(actions)
#     for action_name, method in actions:

