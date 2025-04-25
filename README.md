# CSTR-Fault-Diagnosis-ML
Machine learning-based fault detection system for Continuous Stirred-Tank Reactors (CSTRs) using SVC, Random Forest, and Logistic Regression models, focused on domain adaptation and automated fault diagnosis.

Overview:
This project focuses on building an automated fault-detection system for Continuous Stirred-Tank Reactors (CSTRs) using machine learning techniques. The aim is to classify 12 distinct fault types from sensor data and historical process records, helping to minimize downtime and enhance operational efficiency in chemical engineering processes.

Problem Statement:
Traditional fault detection relies heavily on manual monitoring, which is slow and error-prone. This project addresses the cross-domain fault diagnosis challenge, where models trained on simulated data must accurately detect faults in real-world reactor operations, despite differences in data distributions.

Solution Approach:
Preprocessing 200 minutes of sensor readings (sampled every minute) with imputation, normalization, and optional segmentation.

Training three machine learning models:
Support Vector Classifier (SVC)
Random Forest Classifier
Logistic Regression
Hyperparameter tuning using GridSearchCV.
Model validation using k-fold cross-validation, with performance evaluated through accuracy, precision, recall, F1-score, and ROC analysis.

Data Source:
The dataset was sourced from the Department of Teleinformatics Engineering, Federal University of Ceará, Brazil, following all ethical guidelines and data privacy regulations.

Tools and Technologies:
Programming Language: Python 3
Libraries:
Data Processing: NumPy, Pandas
Visualization: Matplotlib, Seaborn
Machine Learning: Scikit-learn
Parallel Processing: Multiprocessing

Results:
Random Forest Classifier:
Accuracy: ~96.9%
Support Vector Classifier (after tuning):
Accuracy: ~97.6%
Logistic Regression:
Accuracy: ~82.3%
Random Forest and SVC showed high reliability and robustness, while Logistic Regression was limited due to the linear nature of the model.

Challenges Faced:
Cross-domain adaptation complexities.
Limited dataset size (only 200 samples).
Logistic regression struggling with nonlinear relationships and outliers.

Future Work:
Implement more advanced machine learning models (e.g., ensemble methods, deep learning).
Integrate physics-based domain knowledge to enhance model interpretability.
Develop real-time fault detection and adaptive modeling for proactive maintenance.
Expand the dataset to include more fault scenarios and varied operating conditions.
