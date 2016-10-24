import sys
import ConfigParser
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np

class ResourcesFile:
    def __init__(self, current_path, folder_name):
        self.current_path = current_path
        self.folder_name = folder_name
	
    def get(self, filename):
        return self.current_path+'/'+self.folder_name + '/' + filename

def print_name():
    print sys._getframe(1).f_code.co_name

def get_word_tokens(text):
    word_tokens = list()
    for s in sent_tokenize(text):
        word_tokens += word_tokenize(s)
    return word_tokens

def combine_arrays(array1, array2, technique = 'sum'):
    if(technique == 'sum'):
        return np.sum([array1, array2], axis=0)
    else:
        return array1

class AdConfig:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = dict()
        Config = ConfigParser.ConfigParser()
        Config.read(config_file)
        self.config["labelColumn"] = Config.get("TrainingData", "labelColumn")
        self.config["textColumn"] = Config.get("TrainingData", "textColumn")

    def get(self, name):
        return self.config[name]