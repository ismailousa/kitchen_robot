import torch
from torch.utils.data import DataLoader, random_split
from kitchen_robot.components.prepare_model import PredictionModel
from kitchen_robot.entity.config_entity import ModelTrainingConfig


class ModelTraining:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = PredictionModel(self.config.feature_size, self.config.target_size)
        self.model.load_state_dict(torch.load(self.config.base_model_path))

    def train_valid_generator(self):
        loaded_dataset = torch.load(self.config.train_eval_dataset)

        train_size = int(0.9 * len(loaded_dataset))
        eval_size = len(loaded_dataset) - train_size

        train_dataset, eval_dataset = random_split(
            loaded_dataset, [train_size, eval_size]
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        self.eval_loader = DataLoader(eval_dataset, batch_size=self.config.batch_size)

    @staticmethod
    def save_model(path, model):
        torch.save(model.state_dict(), path)

    def train(self):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters())

        for epoch in range(self.config.epochs):
            self.model.train()
            running_loss = 0.0
            running_accuracy = 0.0

            for inputs, targets in self.train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                accuracy = torch.mean(torch.abs(outputs - targets))
                running_accuracy += accuracy.item()

            epoch_loss = running_loss / len(self.train_loader)
            epoch_accuracy = running_accuracy / len(self.train_loader)
            print(f"Epoch {epoch+1}, Loss: {epoch_loss}, Acc: {epoch_accuracy}")

        self.save_model(self.config.updated_model_path, self.model)
        print("Training complete")
