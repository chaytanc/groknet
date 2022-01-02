import os

import torch
from bs4 import BeautifulSoup
from PIL import Image
import numpy as np


class Observer:

    def see(self, driver):
        """
        Get screenshot of browser and return pixel array.
        :param driver: Input the driver which you wish to observe
        :return: If 500x500 resolution screenshot, return 500x500 array.
        """
        tmp = "./screenshot.png"
        driver.save_screenshot(tmp)
        pix_array = np.asarray(Image.open(tmp))
        os.remove(tmp)
        pix_array = torch.Tensor(pix_array)
        return pix_array

    def read_html(self, driver):
        """
        Scrape the html of the current page and return it as a bag of words?? As a sequence of strings broken by spaces?
        :param driver: Input the driver which you wish to observe
        :return: tensor html of the page the driver is on
        """
        soup = BeautifulSoup(driver.page_source)
        #XXX what does this do? How do you turn a string to a tensor??
        # need to map words to growing dictionary structure and then convert to tensor
        # https://discuss.pytorch.org/t/how-to-convert-strings-to-tensors/107507
        html = torch.Tensor(soup)
        return html

    def sense_other_actions(self):
        """
        Get the actions that the other agents have most recently taken
        :return: A tensor encoding the actions that other agents have recently taken
        """