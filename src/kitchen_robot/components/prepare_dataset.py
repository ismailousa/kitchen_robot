import torch
from torch.utils.data import Dataset

import pandas as pd
from sklearn.model_selection import train_test_split

from kitchen_robot.entity.config_entity import PrepareDatasetConfig


class RecipeDataset(Dataset):
    def __init__(self, data):
        self.features = torch.tensor(
            data[
                [
                    "distinct_ingredients",
                    "humidity",
                    "temperature",
                    "taste_rating",
                    "texture_feedback",
                    "user_satisfaction_rating"
                ]
            ].values,
            dtype=torch.float32,
        )
        self.targets = torch.tensor(
            data[["ingredient_quantities", "cooking_time"]].values, dtype=torch.float32
        )
        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class PrepareDataset:
    def __init__(self, config: PrepareDatasetConfig):
        self.texture_mapping = {"soft": 1, "medium": 2, "hard": 3}
        self.config = config

    def get_features(self):
        feedback_data = pd.read_csv(self.config.feedback_data)
        environmental_data = pd.read_csv(self.config.environment_data)
        historical_data = pd.read_csv(self.config.historical_data)

        feedback_data["texture_feedback"] = feedback_data["texture_feedback"].map(
            self.texture_mapping
        )
        feedback_data_avg = feedback_data.groupby("recipe_id").agg(
            {
                "taste_rating": "mean",
                "texture_feedback": "mean",
                "presentation_score": "mean",
            }
        )
        environmental_data_avg = environmental_data.groupby("recipe_id").agg(
            {"humidity": "mean", "temperature": "mean"}
        )

        historical_data = historical_data.merge(
            environmental_data_avg, on="recipe_id", how="left"
        )
        historical_data = historical_data.merge(
            feedback_data_avg, on="recipe_id", how="left"
        )

        train_eval_data, test_data = train_test_split(
            historical_data, test_size=0.1, shuffle=True, random_state=42
        )

        # Save the train_eval_data and test_data to CSV files
        train_eval_data.to_csv(self.config.train_eval_data, index=False)
        test_data.to_csv(self.config.test_data, index=False)

    def generate_dataset(self):
        train_eval_data = pd.read_csv(self.config.train_eval_data)
        test_data = pd.read_csv(self.config.test_data)

        train_eval_dataset = RecipeDataset(train_eval_data)
        test_dataset = RecipeDataset(test_data)

        self._save_dataset(train_eval_dataset, self.config.train_eval_dataset)
        self._save_dataset(test_dataset, self.config.test_dataset)

    def _save_dataset(self, dataset, dataset_path):
        torch.save(dataset, dataset_path)
