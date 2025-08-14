import random


class Config:
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Config, cls).__new__(cls)

            cls.c = 0.3

            cls.RANDOM_SEED = False
            cls.TORCH_DEVICE = 'cpu'
            cls.DATASET_NAME = 'MNIST'
            cls.TEST_SIZE = 0.2
            cls.LABELING_MECHANISM = 'SCAR'
            cls.TRAIN_LABEL_DISTRIBUTION= None
            cls.TEST_LABEL_DISTRIBUTION = None
            cls.SCALE_DATA = "standard" # or "minmax"
        

            cls.EPOCHS= 500
            cls.INITIAL_GUESS_C = 0.3
            cls.LEARNING_RATE = 0.001
            cls.LEARNING_RATE_C = 3* 1 / cls.EPOCHS

            # state variables
            cls.true_prior_proba = None
            cls.train_prior_proba = None
            cls.test_prior_proba = None
            cls.true_train_labels = None
            cls.PU_test_labels = None

        return cls.instance

    @property
    def SEED(self):
        if self.RANDOM_SEED:
            return random.randint(0, 1000000)
        else:
            return 42

    def to_dict(self):
            return {k: v for k, v in self.__class__.__dict__.items() if not k.startswith('_')}

CONFIG = Config()
