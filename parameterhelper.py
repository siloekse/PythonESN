import copy
import json
import logging
from math import ceil
import os, sys

# Check python version (for str/basestring)
if sys.version_info[0] == 3:
    str_type = str,
else:
    str_type = basestring,

# Initialize logger
logger = logging.getLogger(__name__)

# Set paths
CONFIG_PATH = './configs/opt'
USER_PATH = './configs/user'
DEFAULT_CONFIG = 'default'
PARAMETER_FORMAT_CONFIG_PERCENT_DIM = 'parameter_format_percent_dim'
PARAMETER_FORMAT_CONFIG_N_DIM = 'parameter_format_n_dim'

class ParameterHelper(object):
    def __init__(self, filename, percent_dim = False):
        # Choose correct format file
        if percent_dim:
            logger.info('Loading config for dimensionality as a percentage of the reservoir size.')
            parformat_config = PARAMETER_FORMAT_CONFIG_PERCENT_DIM
        else:
            logger.info('Loading config for dimensionality as an integer.')
            parformat_config = PARAMETER_FORMAT_CONFIG_N_DIM

        # Read config files
        default_config = json.load(open(CONFIG_PATH + '/' + DEFAULT_CONFIG + '.json', 'r'))

        # Check if it is a config in the config dir
        use_default_config = False

        # Support both with and without file extension
        if '.json' in filename:
            extension = ''
        else:
            extension = '.json'

        if os.path.exists(USER_PATH + '/' + filename + extension):
            # Package provided config file exists!
            logger.info("Config: %s (overload)"% filename)
            configfile = USER_PATH + '/' + filename + extension

        elif os.path.exists(filename + extension):
            # Custom config file exists!
            logger.info("Config: %s (overload)"% filename)
            configfile = filename + extension

        else:
            # None exist. Use default.
            logger.warning("Could not find the provided config. Using default.")
            use_default_config = True

        # Overload default config file if appropriate
        if not use_default_config:
            user_config = json.load(open(configfile, 'r'))

            # Overload the default config with the one provided by the user.
            self._optconfig = self._overload_config(default_config, user_config)
        else:
            self._optconfig = default_config

        # Read the parameter format
        self._parameter_format = json.load(open(CONFIG_PATH + '/' + parformat_config + '.json', 'r'))

        # Define operators
        self._operators = {"multiply_intreturn": self._multiply_intreturn}

        self._parse()

    def _parse(self):
        # Return {optimization, parameter_format, parameters, sigma}
        self._optimization = self._optconfig['optimization']

        self._fixed_values = dict()
        self._prototype = dict()
        self._sigma = dict()

        ## Embedding
        self._fixed_values['embedding'] = self._optconfig['embedding']
        self._parameter_format['embedding_parameters'] = \
                self._parameter_format['embedding_parameters'][self._optconfig['embedding']]

        # n_dim
        self._parameter_format['n_dim'] = \
                self._parameter_format['n_dim'][self._optconfig['embedding']]

        ## Regression method
        self._fixed_values['regression_method'] = self._optconfig['regression_method']
        self._parameter_format['regression_parameters'] = \
                self._parameter_format['regression_parameters'][self._optconfig['regression_method']]

        ## Get the parameter values/bounds/types from the config
        # to generate prototype.
        for key in self._optconfig:
            if key == 'optimization' or key == 'embedding' or key == 'regression_method':
                continue

            # Check if parameter is necessary
            if not self._need_parameter(self._parameter_format, key):
                continue

            # Constant. Store in fixed values.
            if self._optconfig[key]['type'] == 'c':
                self._fixed_values[key] = self._optconfig[key]['val']
            # Not constant. Add to prototype individual.
            else:
                self._prototype[key] = (self._optconfig[key]['type'],
                                           self._optconfig[key]['min'],
                                           self._optconfig[key]['max'])
                self._sigma[key] = self._optconfig[key]['sigma']

        return

    def get_parameters(self, individual):
        # Copy the parameter format structure and replace with the proper values (from individual and fixed values)
        parameters = copy.deepcopy(self._parameter_format)

        for key in parameters:
            # String => replace with values
            if isinstance(parameters[key], str_type):
                if parameters[key] in self._fixed_values:
                    parameters[key] = self._fixed_values[parameters[key]]
                elif parameters[key] in individual:
                    parameters[key] = individual[parameters[key]]

            # list => replace each string in the list with values
            elif type(parameters[key]) == type([]):
                for i in range(len(parameters[key])):
                    if parameters[key][i] in self._fixed_values:
                        parameters[key][i] = self._fixed_values[parameters[key][i]]
                    elif parameters[key][i] in individual:
                        parameters[key][i] = individual[parameters[key][i]]

            # Dict => This is an operator that needs to be called
            elif type(parameters[key]) == type({}):
                # Should have "operator", "val1" and "val2"
                if parameters[key]["operator"] in self._operators:

                    # Value fixed
                    if parameters[key]["val1"] in self._fixed_values:
                        val1 = self._fixed_values[parameters[key]["val1"]]

                    # Value in individual
                    elif parameters[key]["val1"] in individual:
                        val1 = individual[parameters[key]["val1"]]

                    # Value2 fixed
                    if parameters[key]["val2"] in self._fixed_values:
                        val2 = self._fixed_values[parameters[key]["val2"]]

                    # Value2 in individual
                    elif parameters[key]["val2"] in individual:
                        val2 = individual[parameters[key]["val2"]]

                    parameters[key] = self._operators[parameters[key]["operator"]](val1, val2)

                else:
                    raise ValueError("Invalid operator " + parameters[key]["operator"] + " defined for "+ key)

        return parameters

    def get_prototype(self):
        return self._prototype, self._sigma

    def _overload_config(self, default, overload):
        """
        Overload the default config with the user provided config.
        """
        # Check the provided parameters in overload. Replace the ones in the default config
        # with these parameters.

        config = copy.deepcopy(default)

        for key in overload:
            # dicts need to be iterated over
            if type(overload[key]) == type({}):
                for inner_key in overload[key]:
                    config[key][inner_key] = overload[key][inner_key]
            else:
                config[key] = overload[key]

        return config

    def _need_parameter(self, parameter_format, parameter_name):
        # Check if the current parameter is needed for the current setup
        found_it = False

        for key in parameter_format:
            if parameter_format[key] is None:
                pass

            # List parameters
            elif type(parameter_format[key]) == type([]):
                for i in range(len(parameter_format[key])):
                    if isinstance(parameter_format[key][i], str_type):
                        if parameter_format[key][i] == parameter_name:
                            found_it = True

            # Pure parameters
            elif isinstance(parameter_format[key], str_type):
                if parameter_format[key] == parameter_name:
                    found_it = True

            # For operators (represented as dict)
            elif type(parameter_format[key]) == type({}):
                if parameter_format[key]['val1'] == parameter_name or parameter_format[key]['val2'] == parameter_name:
                    found_it = True

        return found_it

    def _multiply_intreturn(self, val1, val2):
        return int(ceil(val1*val2))

if __name__ == "__main__":
    parhelp = ParameterHelper('nusvr_kpca')
    prototype, sigma = parhelp.get_prototype()
