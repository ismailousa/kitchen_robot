import torch

from kitchen_robot.components.prepare_model import PredictionModel
from kitchen_robot.entity.config_entity import ModelInferenceConfig


class ModelInference:
    def __init__(self, config: ModelInferenceConfig):
        self.config = config
        self.model = None

    def load_model(self):
        self.model = PredictionModel(self.config.feature_size, self.config.target_size)
        self.model.load_state_dict(torch.load(self.config.model_path))
        self.model.eval()

    def infer(self):
        features = torch.tensor(
            [
                self.config.features.distinct_ingredients,
                self.config.features.humidity,
                self.config.features.temperature,
                self.config.features.texture_feedback,
                self.config.features.user_satisfaction_rating,
            ],
            dtype=torch.float32,
        )
        with torch.no_grad():
            output = self.model(features)
        print(output.tolist())
        return output.tolist()
