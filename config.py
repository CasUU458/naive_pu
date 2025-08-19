import random
import json

class Config:
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Config, cls).__new__(cls)

            cls.label_frequency = 0.2
            cls.c = cls.label_frequency

            cls.RANDOM_SEED = False
            cls.TORCH_DEVICE = 'cpu'
            cls.DATASET_NAME = 'BreastCancer' #MNIST, BreastCancer
            cls.TEST_SIZE = 0.2
            cls.LABELING_MECHANISM = 'SCAR'
            cls.TRAIN_LABEL_DISTRIBUTION= None
            cls.TEST_LABEL_DISTRIBUTION = None
            cls.SCALE_DATA = "standard" # or "minmax"


            cls.EPOCHS= 300
            cls.INITIAL_GUESS_C = None
            cls.LEARNING_RATE = 0.001

            cls.LEARNING_RATE_C_modifier = 1
            cls.LEARNING_RATE_C =  cls.LEARNING_RATE_C_modifier*1 / cls.EPOCHS
            cls.penalty = "l2" #None, l2 or "l1"
            cls.solver = 'adam' # lbfgs or adam
            cls.VALIDATION_FRAC = 0.2
            cls.TM_ALPHA = 0.2
            # state variables
            cls.true_prior_proba = None
            cls.train_prior_proba = None
            cls.test_prior_proba = None
            cls.true_train_labels = None
            cls.PU_test_labels = None
            cls.dominant_features = None
        return cls.instance

    @property
    def SEED(self):
        if self.RANDOM_SEED:
            return random.randint(0, 1000000)
        else:
            return 42

    def to_dict(self):
            return {k: v for k, v in self.__class__.__dict__.items() if not k.startswith('_')}
    
    def update_from_dict(self, config_dict: dict):
        """Update config settings from a dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: {key} is not a valid config attribute and will be ignored.")

    def from_json(self, json_path: str):
        """Load config from a JSON file and update current settings."""
        with open(json_path, "r") as f:
            config_dict = json.load(f)
        self.update_from_dict(config_dict)


CONFIG = Config()

