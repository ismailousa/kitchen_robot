## Kitchen Robot Pipelines

### Project Overview

This project showcases a Kitchen Robot system with AI pipelines designed to optimize resource usage and enhance culinary outcomes. This Proof of Concept aims to demonstrate a structured approach for future enhancements and iterative improvements.

### About the Project

Due to time constraints, this project was developed in a single evening to present a foundational structure and prototype. It serves as a starting point that can be iterated upon to enhance quality, scalability, and functionality.

A small sequential model (regression task) that can fit in a resource constrained robot and no connectivity is trained. Model can be retrained on new data.

### Installation and Execution

- Clone the repository to your local machine.
- Install dependencies using Poetry:
  ```
  poetry install
  ```
- Activate the virtual environment:
  ```
  poetry shell
  ```
- Run the main script to execute all pipelines:
  ```
  python main.py
  ```
- For specific pipeline execution:
  ```
  python src/kitchen_robot/pipelines/{pipeline_name}
  ```

### Pipeline Descriptions

- data_ingestion: Ingests historical cooking data, environmental variables, and user feedback.
- prepare_dataset: Prepares the dataset for training and evaluation.
- prepare_model: Prepares the base model for training.
- model_training: Trains the model on the prepared dataset.
- model_evaluation: Evaluates the trained model's performance.
- model_inference: TODO (C++ onnx runtime for robot)

### Project Structure

- artifacts: Contains data, model files, and evaluation reports.
- config: Configuration files for the project.
- research: Jupyter notebooks for exploratory data analysis.
- src: Source code for the Kitchen Robot project.
- tests: Unit tests for the project.

### Summary

This POC demonstrates key AI pipelines: data ingestion, dataset preparation, model training, and evaluation.

The project structure includes artifact storage, configuration files, research notebooks, source code, and unit tests.

Note: Tests, CI/CD, proper training, and additional features are not included due to time constraints.

### Next Steps
- Reiterate feature selection and modeling to improve accuracy
- Enhance the project with testing suites for robustness and reliability.
- Integrate CI/CD pipelines for automated testing and deployment.
- Implement further functionalities and optimizations for an advanced Kitchen Robot system.

This project serves as a starting point for a comprehensive AI solution for optimizing cooking processes and fulfilling culinary expectations. It showcases the potential for expansion and improvement with additional features and quality enhancements.

Feel free to adapt and extend this POC for future development stages.

By Isma-Ilou Sadou