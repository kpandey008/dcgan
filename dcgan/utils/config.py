# Parses the config file used for hyperparameter training
import configparser
import os

class ConfigLoader:
    def __init__(self, path):
        # TODO: Modify the code to account for utomated checking of the file
        # in the root project dir
        if not os.path.isfile(path):
            raise FileNotFoundError('The config file could not be found at the specified path')

        config = configparser.ConfigParser()
        config.read(path)

        self.config = config

    def get_param_value(self, section, parameter=None):
        """
        Returns the hyperparameter corresponding to the title and parameter in
        the config file
        """
        if not parameter:
            # Return the list of all parameters in the section
            return self.config[section]

        return self.config[section][parameter]
