# encoding:utf-8
import logging.config

class Logger:
    def __init__(self, con_path="conf.ini"):
        logging.config.fileConfig(con_path)

    def getLogger(self, name):
        return logging.getLogger(name)
