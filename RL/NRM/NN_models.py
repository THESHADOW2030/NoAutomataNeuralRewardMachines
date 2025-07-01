import torch.nn as nn
import torch.nn.functional as F

sftmx = torch.nn.Softmax(dim=-1)

def sftmx_with_temp(x, temp):
    return sftmx(x / temp)
    
class CNN_grounder(nn.Module):
    def __init__(self, num_symbols):
        super(CNN_grounder, self).__init__()

        self.conv1 = nn.Conv2d(3, 5, kernel_size=5)
        self.conv2 = nn.Conv2d(5, 5, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(125, 50)
        self.fc2 = nn.Linear(50, num_symbols)
        self.softmax = nn.Softmax()

    def forward(self, x, temp=1):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 3))
        # x = F.relu(F.max_pool2d(self.conv2_drop(x), 3))
        x = self.flat(x)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = sftmx_with_temp(x, temp)
        return x

class Linear_grounder_no_droput(nn.Module):
    def __init__(self, num_inputs, hidden_size, num_output):
        super(Linear_grounder_no_droput, self).__init__()
        self.grounder = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softmax(),
            nn.Linear(hidden_size, num_output),
        )
    def forward(self, x):
         return self.grounder(x)

class Linear_grounder(nn.Module):
    def __init__(self, num_inputs, hidden_size, num_output):
        super(Linear_grounder, self).__init__()
        self.grounder = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Dropout(0.2),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_output),
        )
    def forward(self, x, temp = 1):
        x = self.grounder(x)
        return sftmx_with_temp(x, temp)
