import random


class Config:
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Config, cls).__new__(cls)

            cls.c = 0.5

            cls.RANDOM_SEED = True
            cls.TORCH_DEVICE = 'cpu'
            cls.DATASET_NAME = 'MNIST'
            cls.TEST_SIZE = 0.2

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


CONFIG = Config()
