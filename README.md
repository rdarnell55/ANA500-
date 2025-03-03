# ANA500 Alzheimer's Prediction - Course Project

## Project Overview
This Jupyter Notebook is the course project for ANA500: Python for Data Analysis. It focuses on predicting Alzheimer's disease onset using data analysis and machine learning techniques. The project explores various risk factors, visualizations, and predictive modeling approaches.

Overall Findings: The Random Forest Classifier demonstrated strong predictive power among the ML models, while LSTM and GRU performed well in the deep learning approach, showing potential for capturing complex temporal patterns. The deep learning models slightly outperformed traditional ML models in accuracy and recall, making them promising for early detection of Alzheimer's onset.

## Files and Structure
- **ANA500 Course Project.ipynb**: The main Jupyter Notebook containing all analysis, visualizations, and predictive modeling.
- **Dataset**: `alzheimers_risk_factors.csv` - The dataset used in this project, containing demographic, lifestyle, medical, and genetic variables related to Alzheimer's disease risk.
- **Data Files**: The notebook expects the dataset to be located in the appropriate directory.

## Dependencies
The following Python libraries are used in the notebook:
- `pandas` (for data manipulation)
- `numpy` (for numerical computations)
- `matplotlib` and `seaborn` (for data visualization)
- `scikit-learn` (for machine learning models)
- `statsmodels` (for statistical analysis)
- `tensorflow` and `keras` (for deep learning model implementation)

## Usage
1. Ensure all dependencies are installed.
2. Open the Jupyter Notebook (`ANA500 Course Project.ipynb`).
3. Run each cell sequentially to reproduce the analysis.
4. Modify or extend the analysis based on specific research questions.

## Data Cleaning Steps
- **Data Preprocessing**: The dataset undergoes preprocessing, including handling missing values, feature engineering, and exploratory data analysis.

## Machine Learning Models Applied
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Random Forest Classifier**

## Deep Learning Models Applied
- **Recurrent Neural Network (RNN)**
- **Long Short-Term Memory (LSTM)**
- **Gated Recurrent Unit (GRU)**
- **Sequential Model with Dense Layers**

## Model Performance Functions
1. **Data Splitting**: The dataset was divided into training and testing sets.
2. **Feature Scaling**: Standardization techniques were applied to ensure consistency across models.
3. **Model Training**: Each machine learning and deep learning model was trained on the training dataset.
4. **Evaluation Metrics**: Models were assessed using accuracy, precision, recall, F1-score, and AUC-ROC.
5. **Hyperparameter Tuning**: Grid search and random search techniques were used to optimize model parameters.
6. **Performance Visualization**: Results were compared through confusion matrices, precision-recall curves, and ROC curves.

## Visualization Functions
1. **Feature Distribution Plots**: Histograms and density plots were used to analyze data distribution.
2. **Correlation Heatmap**: Visual representation of feature relationships using a heatmap.
3. **Pair Plots**: Scatterplot matrices to observe trends between variables.
4. **Box Plots**: Used for identifying outliers and understanding data spread.
5. **Confusion Matrix**: Graphical representation of model classification performance.
6. **ROC and Precision-Recall Curves**: Used to evaluate model discrimination ability and balance between precision and recall.

## Future Improvements
- Incorporation of deep learning models.
- Expansion of dataset to include additional risk factors.
- Optimization of feature selection and hyperparameter tuning.

## Acknowledgments
This project is part of the ANA500 course at National University, focusing on data analytics using Python.
