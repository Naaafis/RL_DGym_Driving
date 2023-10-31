import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, action_size, device):
        """ Create Q-network
        Parameters
        ----------
        action_size: int
            number of actions
        device: torch.device
            device on which to the model will be allocated
        """
        super().__init__()

        self.device = device 
        self.action_size = action_size

        # TODO: Create network
    
        # Convolutional layers to process the raw pixel observations
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Fully connected layers after extracting sensor values
        self.fc1 = nn.Linear(4096 + 7, 512)  # 7*7*64 from conv layers + 7 from sensor values (1 speed, 4 abs, 1 steering, 1 gyro)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, observation):
        """ Forward pass to compute Q-values
        Parameters
        ----------
        observation: np.array
            array of state(s)
        Returns
        ----------
        torch.Tensor
            Q-values  
        """

        # TODO: Forward pass through the network
        batch_size = observation.shape[0]
        
        # Extract sensor values from the observation
        speed, abs_sensors, steering, gyroscope = self.extract_sensor_values(observation, batch_size)

        # reorder from (batch_size, height, width, channels) to (batch_size, channels, height, width)
        observation = observation.permute(0, 3, 1, 2)

        
        # # Process raw pixel observations through convolutional layers
        # x = F.relu(self.conv1(observation))
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # x = x.reshape(x.size(0), -1)  # Flatten
        
        # # Concatenate with the extracted sensor values
        # x = torch.cat([x, speed, abs_sensors, steering, gyroscope], dim=1)
        
        # # Pass through fully connected layers
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)

        x = F.relu(self.conv1(observation))
        #print(f"Shape after conv1: {x.shape}")
        
        x = F.relu(self.conv2(x))
        #print(f"Shape after conv2: {x.shape}")

        x = F.relu(self.conv3(x))
        #print(f"Shape after conv3: {x.shape}")

        x = x.reshape(x.size(0), -1)  # Flatten
        #print(f"Shape after flatten: {x.shape}")

        x = torch.cat([x, speed, abs_sensors, steering, gyroscope], dim=1)
        #print(f"Shape after concatenating with sensor values: {x.shape}")

        x = F.relu(self.fc1(x))
        #print(f"Shape after fc1: {x.shape}")

        x = self.fc2(x)
        #print(f"Shape after fc2: {x.shape}")


        return x


    def extract_sensor_values(self, observation, batch_size):
        """ Extract numeric sensor values from state pixels
        Parameters
        ----------
        observation: list
            python list of batch_size many torch.Tensors of size (96, 96, 3)
        batch_size: int
            size of the batch
        Returns
        ----------
        torch.Tensors of size (batch_size, 1),
        torch.Tensors of size (batch_size, 4),
        torch.Tensors of size (batch_size, 1),
        torch.Tensors of size (batch_size, 1)
            Extracted numerical values
        """
        # print("observation shape: ", observation.size())
        speed_crop = observation[:, 84:94, 12, 0].reshape(batch_size, -1)
        speed = speed_crop.sum(dim=1, keepdim=True) / 255
        abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
        abs_sensors = abs_crop.sum(dim=1) / 255
        steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1)
        steering = steer_crop.sum(dim=1, keepdim=True)
        gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1)
        gyroscope = gyro_crop.sum(dim=1, keepdim=True)
        return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope