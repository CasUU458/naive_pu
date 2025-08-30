import random

class Config:
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Config, cls).__new__(cls)

            cls.c = 0.2 # Labeling frequency

            # cls.RANDOM_SEED = False
            cls.device = 'cpu'
            cls.dataset = 'mock' #MNIST, BreastCancer #mock or diabetes
            cls.test_size = 0.2
            cls.label_mechanism = 'SCAR_1_100'
            cls.positive_ratio = None # number of positives / number of negatives or None to ignore
            cls.scaler = "standard" # or "minmax"


            cls.max_iterations= 300
            cls.naive_c_guess = None
            cls.lr = 0.005

            cls.penalty = "l2" #None, l2 or "l1"
            cls.solver = 'adam' # lbfgs or adam

            cls.random_state = 42
            cls.tolerance = 1e-4

            #Naive model parameters
            cls.lr_c =  0.005
    

            # Two model parameters
            cls.epsilon = 1e-4
            cls.alpha = None
            cls.max_loop_iterations = 100
            cls.validation_frac = None

            # state variables
            cls.true_prior_proba = None
            cls.train_prior_proba = None
            cls.test_prior_proba = None
            cls.true_train_labels = None
            cls.PU_test_labels = None
            cls.dominant_features = None
            cls.SAR_c_log = None
        return cls.instance

    # @property
    # def SEED(self):
    #     if self.RANDOM_SEED:
    #         return random.randint(0, 1000000)
    #     else:
    #         return 42

    def to_dict(self):
            return {k: v for k, v in self.__class__.__dict__.items() if not k.startswith('_')}
    
    # def update_from_dict(self, config_dict: dict):
    #     """Update config settings from a dictionary."""
    #     for key, value in config_dict.items():
    #         if hasattr(self, key):
    #             setattr(self, key, value)
    #         else:
    #             print(f"Warning: {key} is not a valid config attribute and will be ignored.")

    # def from_json(self, json_path: str):
    #     """Load config from a JSON file and update current settings."""
    #     with open(json_path, "r") as f:
    #         config_dict = json.load(f)
    #     self.update_from_dict(config_dict)

    def set_random_seed(self,seed=None):
        if seed is not None:
            self.random_state = seed
        else:
            self.random_state = random.randint(0, 1000000)

    def set_attr(self, attr_name, attr_value):
        if hasattr(self, attr_name):
            setattr(self, attr_name, attr_value)
        else:
            print(f"Warning: {attr_name} is not a valid config attribute and will be ignored.")

    def get_attr(self, attr_name):
        if hasattr(self, attr_name):
            return getattr(self, attr_name)
        else:
            print(f"Warning: {attr_name} is not a valid config attribute.")
            return None

CONFIG = Config()

