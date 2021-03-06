"""
Created on 3rd Jan 2017,
@author : Shreesha N
"""

import datetime
import logging

from covid_19.utils.singleton import Singleton


class Logger:  # metaclass=Singleton
    def __init__(self, name=None, log_path=None):
        if name is None or log_path is None:
            raise Exception('Logger name or Logger filepath missing')
        self.logger = logging.getLogger(name=name)
        self.logger.setLevel(logging.INFO)

        timestamp = '{:%Y-%m-%d-%H_%M_%S}'.format(datetime.datetime.now())
        self.file_log_handler = logging.FileHandler(log_path + '/' + name + '.log')
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.file_log_handler.setFormatter(self.formatter)
        self.file_log_handler.setLevel(logging.INFO)
        self.logger.addHandler(self.file_log_handler)

        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(logging.INFO)
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.console_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.console_handler)

    def get_logger(self):
        return self.logger
