from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from kitchen_robot.components.prepare_model import PredictionModel
from kitchen_robot.entity.config_entity import ModelEvaluationConfig
from kitchen_robot.utils.common import save_json


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def load_model(self):
        self.model = PredictionModel(self.config.feature_size, self.config.target_size)
        self.model.load_state_dict(torch.load(self.config.updated_model_path))

    def _load_generator(self):
        dataset = torch.load(self.config.test_dataset)

        self.test_loader = DataLoader(dataset, batch_size=self.config.batch_size)

    def save_score_report(self):
        report = {"loss": self.test_loss}
        save_json(path=Path(self.config.model_performance_report), data=report)

    def evaluate(self):
        self.model.eval()
        self._load_generator()
        self.test_loss = 0.0
        self.test_accuracy = 0.0

        criterion = torch.nn.MSELoss()

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                print(f"inputs {inputs}")
                print(f"targets {targets}")
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                self.test_loss += loss.item()

                accuracy = torch.mean(torch.abs(outputs - targets))
                self.test_accuracy += accuracy.item()

        self.test_loss /= len(self.test_loader)
        self.test_accuracy /= len(self.test_loader)
        print(
            f"Test Loss: {self.test_loss}  for model {self.config.updated_model_path}"
        )
        self.save_score_report()
