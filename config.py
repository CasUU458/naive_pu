import random


class Config:
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Config, cls).__new__(cls)


            cls.C = .5  # Fraction of positives which will be labeled.

            cls.RANDOM_SEED = 42
            cls.TORCH_DEVICE = 'cpu'
            cls.DATASET_NAME = 'BreastCancer'
            cls.TEST_SIZE = 0.2

            cls.LABELING_MECHANISM = "SCAR"
            cls.TRAIN_LABEL_DISTRIBUTION = None
            cls.TEST_LABEL_DISTRIBUTION = None
            cls.SCALE_DATA = "standard"

            #classifier
            cls.PENALTY = None # Default penalty for logistic regression
            cls.SOLVER = "adam"  # lbfgs or adam
            cls.EPOCHS = 300
            cls.TOLERANCE = 1e-6
            cls.INITIAL_GUESS_C = 0.5

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

