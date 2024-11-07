# Industrial Steam Volume Prediction

This project aims to predict industrial steam volume based on boiler sensor data, originally provided as part of a machine learning competition on the Alibaba Tianchi platform.
This is my first machine learning project, focusing on predicting industrial steam volume based on boiler sensor data from the Alibaba Tianchi platform. This project is organized into four main parts, each contained in a Jupyter notebook:

1. **Data Exploration**
2. **Feature Engineering**
3. **Model Predictions**
4. **Model Fusion**


## Project Background

In thermal power generation, steam production is crucial for generating electricity. The basic principle involves burning fuel to heat water, producing high-temperature and high-pressure steam. This steam drives a turbine, which powers an electricity generator. Boiler combustion efficiency is key to overall power generation efficiency. Many factors influence this efficiency, including adjustable boiler parameters and boiler operating conditions.

### Key Parameters Affecting Boiler Efficiency

- **Adjustable Parameters**: Fuel feed rate, primary and secondary air, induced draft, return air, and feedwater flow.
- **Operating Conditions**: Bed temperature, bed pressure, furnace temperature, furnace pressure, and superheater temperature.

## Problem Description

The dataset includes sensor data from a boiler system, anonymized for privacy. The goal is to predict the amount of steam generated based on the operating conditions of the boiler.

### Data Details

The data is split into:
- **Training Data** (`train.txt`): Contains 38 feature fields (V0-V37) and a target field (steam volume).
- **Testing Data** (`test.txt`): Contains only the feature fields (V0-V37), for which predictions are required.

### Evaluation Metric

The model's performance is evaluated based on **Mean Square Error (MSE)** between the predicted and actual steam volume in the test set.

## Submission Format

Participants are required to submit a text file containing the predicted values for the test data. The file should contain a single column representing the predicted steam volume.

## Getting Started

### Prerequisites

- Python 3.x
- Libraries: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `lightgbm`, `matplotlib`, `scipy`, `seaborn`

### Installation

To set up the environment, install the dependencies using the following command:

```bash
pip install numpy pandas scikit-learn xgboost lightgbm
