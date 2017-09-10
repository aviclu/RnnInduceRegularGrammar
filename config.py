import configparser
import logging


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Config(object):
    __metaclass__ = Singleton

    def __init__(self):
        self.__configfile = configparser.ConfigParser()
        self.__configfile.read('config.cfg')
        self.__sections = {}
        self.logger = logging.getLogger('stack_rnn.%s' % str(self.__class__))
        self.logger.info('creating an instance of %s' % str(self.__class__))

    def __getattr__(self, section):
        if section not in self.__sections:
            self.__sections[section] = Config.Section(self.__configfile, section)
        return self.__sections[section]

    class Section(object):
        def __init__(self, config, section_name):
            self.__config = config
            self.__section_name = section_name

        def __getattr__(self, item):
            return Config.Value(self.__config.get(self.__section_name, item))

    class Value(object):
        def __init__(self, value):
            self.value = value

        @property
        def boolean(self):
            return self.value in ('True', 'true')

        @property
        def int(self):
            return int(self.value)

        @property
        def float(self):
            return float(self.value)

        @property
        def str(self):
            return self.value

        @property
        def lst(self):
            return self.value.split(', ')
