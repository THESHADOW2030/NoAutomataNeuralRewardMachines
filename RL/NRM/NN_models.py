import torch.nn as nn
import torch.nn.functional as F
import torch


sftmx = torch.nn.Softmax(dim=-1)

def sftmx_with_temp(x, temp):
    return sftmx(x / temp)
    
class CNN_grounder_old(nn.Module):
    def __init__(self, num_symbols):
        super(CNN_grounder, self).__init__()
        
        # Increased channels from 5 -> 32/64 to capture features like 'color', 'shape', 'edges'
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Less aggressive pooling to preserve small objects (gems/lava)
        self.pool = nn.MaxPool2d(2, 2) 
        self.dropout = nn.Dropout(0.3)
        
        # Calculate flatten size: 
        # 64x64 -> pool -> 32x32 -> pool -> 16x16 -> pool -> 8x8
        # 64 channels * 8 * 8 = 4096
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_symbols)

    def forward(self, x):
        # Layer 1
        x = self.pool(F.relu(self.conv1(x))) # 64 -> 32
        # Layer 2
        x = self.pool(F.relu(self.conv2(x))) # 32 -> 16
        # Layer 3
        x = self.pool(F.relu(self.conv3(x))) # 16 -> 8
        
        x = x.view(-1, 64 * 8 * 8) # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Note: Do NOT apply Softmax here if you use CrossEntropy or Gumbel Softmax later
        # The Gumbel function expects raw logits.
        return x
    

    
    
class CNN_grounder(nn.Module):
    def __init__(self, num_symbols):
        super(CNN_grounder, self).__init__()
        
        # 1. Use 32 -> 64 -> 64 Channels (Enough for this complexity)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) # Added BN from 'New' code
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) # Added BN
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64) # Added BN

        self.pool = nn.MaxPool2d(2, 2) 
        self.dropout = nn.Dropout(0.3)
        
        # 2. CRITICAL: Use Flattening (from 'Old' code)
        # 64x64 -> 32x32 -> 16x16 -> 8x8 spatial size
        self.flat_size = 64 * 8 * 8 
        
        self.fc1 = nn.Linear(self.flat_size, 128)
        self.fc2 = nn.Linear(128, num_symbols)

    def forward(self, x):
        # Apply Conv -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten preserves the "island" of color in the sea of grey
        x = x.view(-1, self.flat_size) 
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
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
        return x
        #return sftmx_with_temp(x, temp)
