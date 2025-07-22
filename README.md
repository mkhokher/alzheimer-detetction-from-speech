# Alzheimer’s Detection from Speech

## Project Overview

This project detects early signs of **Alzheimer's disease** using machine learning models trained on features extracted from **speech recordings**. By analyzing speech patterns like rhythm, silence, pitch, and energy, the system aims to classify whether a speaker shows symptoms of cognitive decline.

The project uses:

* Classic ML models (Voting, XGBoost, Stacking)
* A custom Neural Network built with PyTorch
* Evaluation with metrics like F1-score, AUC-ROC, and Accuracy

## Workflow

### 1. Load and Prepare Data

* Combines `train_data.txt` and `test_data.txt`
* Drops irrelevant columns (like subject ID)
* Scales features using `StandardScaler`

### 2. Split Dataset

* Uses `train_test_split` with stratified labels (80/20)

### 3. Train Models

* **Voting Classifier**: Combines SVM, Random Forest, and KNN
* **Gradient Boosting**: XGBoost with hyperparameter tuning
* **Stacking Classifier**: Stacks multiple models with Logistic Regression as final estimator
* **Neural Network**: A 3-layer fully connected network trained with BCEWithLogitsLoss

### 4. Evaluate Models

* Metrics used:

  * Accuracy
  * F1 Score
  * AUC-ROC
  * Confusion Matrix and Classification Report

### 5. Visualize Results

* A lollipop chart compares performance across models for all three metrics


## Future Improvements

* Add text-based linguistic features (if transcripts are available)
* Improve neural network performance (e.g., deeper architecture, better loss functions)
* Deploy as a web app using Streamlit or Flask

## References

* Qiao et al., 2020 – Speech pauses and Alzheimer’s ([Link](https://pubmed.ncbi.nlm.nih.gov/32250297/))
* García-Gutiérrez et al., 2024 – Audio-based diagnosis techniques ([Link](https://pubmed.ncbi.nlm.nih.gov/38308366/))
* ADDReSS Challenge – Multimodal Alzheimer’s detection benchmark ([Link](https://arxiv.org/abs/2008.04617))

## Acknowledgments

Thanks to the professors, researchers, dataset providers, and open-source contributors who made this project possible.

