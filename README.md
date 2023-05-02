# Water Quality Prediction Using Machine Learning

This repository contains the source code and data for the research paper "Water Quality Prediction Using Machine Learning: A Case Study on Random Forest Algorithm". The primary objective of this study is to develop a machine learning model for predicting water quality parameters based on the Random Forest algorithm.

## Table of Contents
- [Introduction](#introduction)
- [Data](#data)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributors](#contributors)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Introduction

Water quality is a critical aspect of environmental protection, public health, and sustainable development. Machine learning techniques have the potential to improve the accuracy, efficiency, and scalability of water quality monitoring by identifying patterns and relationships in large and complex datasets, predicting water quality parameters, and providing insights to inform decision-making.

This project aims to explore the potential of machine learning in water quality assessment, focusing on the Random Forest algorithm, an ensemble learning method that combines multiple decision trees to make more accurate and robust predictions.

## Data

The dataset used in this study is derived from the [Kaggle Water Potability dataset](https://www.kaggle.com/adityakadiwal/water-potability). The dataset contains information on various water quality parameters, such as pH, hardness, solids, chloramines, sulfate, conductivity, organic carbon, trihalomethanes, and turbidity. The target variable is water potability, a binary variable indicating whether the water is safe for human consumption or not.

## Installation

To set up the project on your local machine, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/your-username/water-quality-prediction.git
   ```

2. Change to the project directory:
   ```
   cd water-quality-prediction
   ```

3. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Jupyter Notebook `water_quality_prediction.ipynb` to train and evaluate the Random Forest model on the dataset.

2. Modify the notebook as needed to experiment with different machine learning algorithms, hyperparameters, or preprocessing techniques.

3. Analyze the results and compare the performance of different models using various evaluation metrics.

## Results

Our study demonstrated the effectiveness of the Random Forest algorithm in predicting water quality parameters with high accuracy. The model's performance was evaluated using various metrics, such as accuracy, precision, recall, and F1 score.

For more details on the results and their implications, refer to the research paper.

## Contributors

- [Ossama Essfadi 🖋 ](https://github.com/essfadi)

## Acknowledgements

We would like to express our deepest gratitude to our supervisor, Doctor Asmae Mourhir, for her invaluable guidance, support, and encouragement throughout the course of this research.

We are also grateful to the [Kaggle Water Potability dataset](https://www.kaggle.com/adityakadiwal/water-potability) contributors for providing the data used in this study.

## License

This project is licensed under the [MIT License](LICENSE).
