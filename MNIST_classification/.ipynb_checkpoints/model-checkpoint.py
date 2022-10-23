import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet,self).__init__()
        self.line1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.line2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax()
        
    def forward(self,x):
        out = self.line1(x)
        out = self.relu(out)
        out = self.line2(out)
        out = self.softmax(out)
        return out
    
    
