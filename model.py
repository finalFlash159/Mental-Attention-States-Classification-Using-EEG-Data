import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class EEGDataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = signals 
        self.labels = labels
        
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        return signal, label

class DeepConvLSTM(nn.Module):
    def __init__(self, input_channels=14, window_size=128*30, lstm_hidden_size=128, lstm_layers=1, num_classes=3):
        super(DeepConvLSTM, self).__init__()

        # Convolution Layer 1 + MaxPooling 
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=32, stride=1, padding=0)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Convolution Layer 2 + MaxPooling
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=16, stride=1, padding=0)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Convolution Layer 3 + MaxPooling
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8, stride=1, padding=0)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Calculate output size after Conv1D + Pooling layers
        conv_output_size = self.calculate_conv_output(window_size)

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=128, # Number of channels from last CNN
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )

        # Fully Connected Layers
        self.fc1 = nn.Linear(lstm_hidden_size, 128)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout2 = nn.Dropout(0.3)

    def calculate_conv_output(self, input_size):
        """
        Calculate output size after each Conv1D + Pooling layer
        """
        size = input_size
        size = (size - 32) // 4 + 1 # Conv1  
        size = size // 4            # Pool1
        size = (size - 16) // 2 + 1 # Conv2
        size = size // 4            # Pool2
        size = (size - 8) // 1 + 1  # Conv3
        size = size // 2            # Pool3
        return size

    def forward(self, x):
        # Convolution 1 + ReLU + Pooling
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        # Convolution 2 + ReLU + Pooling  
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # Convolution 3 + ReLU + Pooling
        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        # Reshape for LSTM (batch_size, seq_len, channels)
        x = x.permute(0, 2, 1)
        
        # LSTM
        lstm_out, _ = self.lstm(x)

        # Get output from last timestep
        last_timestep = lstm_out[:, -1, :]
        
        # Fully Connected Layer 1 + Dropout
        x = F.relu(self.fc1(last_timestep))
        x = self.dropout1(x)
        
        # Fully Connected Layer 2 + Dropout
        x = self.fc2(x)
        x = self.dropout2(x)

        return F.log_softmax(x, dim=1)