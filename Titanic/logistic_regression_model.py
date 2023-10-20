import torch

class LogisticRegressionModel(torch.nn.Module):
 
    def __init__(self):
        super(LogisticRegressionModel, self).__init__() 
        self.layer_linear = torch.nn.Linear(10, 1) # Ten in and one out linear layer
 
    def forward(self, x):
        y_pred_linear = self.layer_linear(x)
        y_pred_sigmoid = torch.sigmoid(y_pred_linear) # Logistic activation function
        return y_pred_sigmoid