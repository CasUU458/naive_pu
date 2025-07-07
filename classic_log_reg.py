import torch 
import numpy as np


class ClassicLogReg:
    def __init__(self, learning_rate=0.001, num_iterations=1000,device='cpu',tolerance=1e-6):
        
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.device = device
        self.weights: torch.Tensor | None = None
        self.bias: torch.Tensor | None = None
        self.tolerance = tolerance
        self.loss_log = None


    @staticmethod
    def _sigmoid(z):

        #numerical stable stigmoid function
        # positive_mask = z >= 0
        # negative_mask = ~positive_mask

        # out = torch.zeros_like(z, dtype=torch.float32)
        # out[positive_mask] = 1. / (1. + torch.exp(-z[positive_mask]))
        # out[negative_mask] = torch.exp(z[negative_mask]) / (1. + torch.exp(z[negative_mask]))


        out = 1. / (1. + torch.exp(-z))

        return out


    #loss function binary cross entropy or log likelihood?
    def _loss (self, y_true, y_pred):

        eps = 1e-15  # to avoid log(0), numerical stability, small constant
        # ensure y_pred is in the range [eps, 1-eps]
        #clamp_min y_pred to avoid log(0)

        #positive contribution
        y_pred = y_pred.clamp(eps, 1. - eps)
        term1 = y_true* torch.log(y_pred)
        term2 = (1. - y_true) * torch.log(1. - y_pred)
        loss = -torch.mean(term1 + term2)
        return loss
    

    def fit(self, X, y):
        num_samples, n_features = X.shape
        # self.weights = torch.zeros((num_features,1), dtype=torch.float32,requires_grad=True)
        # self.bias = torch.zeros(1, dtype=torch.float32,requires_grad=True)

        # y_t = torch.as_tensor(y, dtype=torch.float32)

        self.weights = torch.zeros(n_features, device=self.device, requires_grad=True)
        self.bias = torch.zeros(1, device=self.device, requires_grad=True)

        # Convert labels to 1-D
        X_t = torch.as_tensor(X, dtype=torch.float32,device=self.device)

        y_t = torch.as_tensor(y, dtype=torch.float32, device=self.device)


        self.optimizer = torch.optim.Adam([self.weights, self.bias], lr=self.learning_rate)
        
        prev_loss = float('inf')

        self.loss_log = np.zeros(self.num_iterations)

        for _ in range(self.num_iterations):
            linear_model = X_t @ self.weights + self.bias
            y_predicted = self._sigmoid(linear_model)
            if _ % 100 == 0:
                print(f"Iteration {_}, Loss: {self._loss(y_t, y_predicted).item()}")

            # loss = torch.nn.functional.binary_cross_entropy(y_predicted, y_t,reduction='mean')
            loss = self._loss(y_t, y_predicted)
            #reset the gradients to zero
            self.optimizer.zero_grad()            
            ## calculate gradients
            loss.backward()
            #update the weights and bias
            self.optimizer.step()

            #Log the loss progression
            self.loss_log[_] = loss.item()

            # check for convergence
            if abs(prev_loss - loss.item()) < self.tolerance:
                print(f"Converged after {_} iterations")
                break   

        return self



    def predict(self, X,th=0.5):
        X = torch.tensor(X, dtype=torch.float32)
     
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been trained yet. Call fit() before predict().")
     

        linear_model = X @ self.weights + self.bias
        y_predicted = self._sigmoid(linear_model).detach().numpy()
        return np.array([1 if i > th else 0 for i in y_predicted])

    def predict_proba(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been trained yet. Call fit() before predict().")
        
        linear_model = X @ self.weights + self.bias
        y_predicted = self._sigmoid(linear_model).detach().numpy()
        return y_predicted

    def get_loss_log(self):
        if self.loss_log is None:
            raise ValueError("Loss log is not available. Make sure to call fit() first.")
        return self.loss_log

    def get_weights(self):
        if self.weights is None:
            raise ValueError("Model has not been trained yet. Call fit() before get_weights().")
        return self.weights.detach().cpu().numpy()

    def get_bias(self):
        if self.bias is None:
            raise ValueError("Model has not been trained yet. Call fit() before get_bias().")
        return self.bias.detach().cpu().numpy()