# EL_task_5

This Task demonstrates the application of tree-based machine learning models 'Decision Trees and Random Forestst' to predict the presence of heart disease using the heart.csv dataset.
## Overview
The goal of this project is to:
Explore the heart disease dataset
Build and visualize a Decision Tree classifier
Analyze overfitting and the effect of tree depth
Train and evaluate a Random Forest classifier
Interpret feature importances
Assess model performance using cross-validation

## Workflow Summary
Data Exploration: Load and inspect the dataset for structure and missing values.
Preprocessing: Separate features and target variable.
Model Training: Train Decision Tree and Random Forest classifiers.
Visualization: Visualize the Decision Tree and feature importances.
Model Evaluation: Compare models using accuracy and cross-validation.

## Key Steps
1. Data Loading & Exploration
Loaded heart.csv and checked for missing values and data distribution.
Ensured data quality before modeling.
2. Data Preparation
Split the data into features (X) and target (y).
Performed an 80-20 train-test split for unbiased model evaluation.
3. Decision Tree Classifier
Trained a Decision Tree on the training set.
Visualized the tree structure using Graphviz.
Evaluated accuracy on both training and test sets.
4. Overfitting Analysis & Depth Control
Trained additional Decision Trees with max_depth=3 and max_depth=5.
Compared their performance to the unrestricted tree to demonstrate the impact of tree complexity on overfitting.
5. Random Forest Classifier
Trained a Random Forest (100 trees) for more robust predictions.
Compared its accuracy to Decision Trees.
6. Feature Importance
Extracted and visualized feature importances from the Random Forest.
Identified which patient features most influence heart disease prediction.
7. Cross-Validation
Performed 5-fold cross-validation on the Random Forest.
Reported mean accuracy and standard deviation for robust performance estimation.

## Results
Decision Trees can easily overfit, but controlling depth improves generalization.
Random Forests outperform single Decision Trees on test accuracy and are less prone to overfitting.
Feature importance analysis reveals key predictors of heart disease.
Cross-validation confirms the stability and reliability of the Random Forest model.
