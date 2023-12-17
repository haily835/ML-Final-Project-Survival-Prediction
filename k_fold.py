import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


def k_fold(X, y, classifier, k_folds=10):
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    smote = SMOTE(random_state=42)

    weighted_accuracies = []

    i = 0
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Apply SMOTE on the training set
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        # Train the classifier on the resampled data
        classifier.fit(X_resampled, y_resampled)

        # Make predictions on the test set
        y_pred = classifier.predict(X_test)

        # Calculate and store the weighted accuracy
        weighted_accuracy = accuracy_score(y_test, y_pred, sample_weight=None)  # You may need to adjust the sample_weight parameter
        weighted_accuracies.append(weighted_accuracy)
        print(f"Fold {i+1}: {weighted_accuracy}")
        i += 1
        

    # Calculate the average weighted accuracy
    average_weighted_accuracy = np.mean(weighted_accuracies)
    print("Average Weighted Accuracy:", average_weighted_accuracy)
    return average_weighted_accuracy