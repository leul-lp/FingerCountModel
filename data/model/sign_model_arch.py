import torch
import torch.nn as nn
import torch.nn.functional as F


class SignModel(nn.Module):
    def __init__(self, input_shape=63, dropout_rate=0.5):
        super(SignModel, self).__init__()
        # Layer 1
        self.fc1 = nn.Linear(input_shape, 256) # Increased neurons
        self.dropout1 = nn.Dropout(dropout_rate)

        # Layer 2
        self.fc2 = nn.Linear(256, 128) # Increased neurons
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Layer 3
        self.fc3 = nn.Linear(128, 64) # Added a new layer with more neurons
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Layer 4
        self.fc4 = nn.Linear(64, 32) # Added a new layer with more neurons
        self.dropout4 = nn.Dropout(dropout_rate)

        # Output Layer
        self.fc5 = nn.Linear(32, 6) # Output layer

    def forward(self, x):
        # Pass through the layers with ReLU and Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        
        x = F.relu(self.fc4(x))
        x = self.dropout4(x)
        
        # Output layer with softmax
        x = F.softmax(self.fc5(x), dim=1) # dim=1 is important for multi-class classification

        return x
