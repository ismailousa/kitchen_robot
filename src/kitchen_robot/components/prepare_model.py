import torch
import torch.nn as nn
import torch.optim as optim


from kitchen_robot.entity.config_entity import PrepareModelConfig


class PredictionModel(nn.Module):
    def __init__(self, feature_size: int, target_size: int):
        super(PredictionModel, self).__init__()
        self.fc1 = nn.Linear(feature_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, target_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PrepareModel:
    def __init__(self, config: PrepareModelConfig):
        self.config = config

    def generate_base_model(self):
        self.model = PredictionModel(self.config.feature_size, self.config.target_size)
        torch.save(self.model.state_dict(), self.config.base_model_path)
