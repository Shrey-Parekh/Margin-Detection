import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_curve, auc

margin_features = pd.read_csv(r'D:\Margin-Detection\margin_labels.csv')
personality_traits = pd.read_csv(r'D:\Margin-Detection\predictions.csv')

margin_features.reset_index(drop=True, inplace=True)
personality_traits.reset_index(drop=True, inplace=True)


X_train, X_test, y_train, y_test = train_test_split(margin_features, personality_traits, test_size=0.2, random_state=42)

nb_model = BernoulliNB()
multi_output_model = MultiOutputClassifier(nb_model)

multi_output_model.fit(X_train, y_train)

y_pred = multi_output_model.predict(X_test)

metrics = {
    'Trait': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1-score': [],
    'Support': []
}

for i, column in enumerate(y_test.columns):
    accuracy = accuracy_score(y_test.iloc[:, i], y_pred[:, i])
    precision = precision_score(y_test.iloc[:, i], y_pred[:, i], zero_division=0)
    recall = recall_score(y_test.iloc[:, i], y_pred[:, i])
    f1 = f1_score(y_test.iloc[:, i], y_pred[:, i])
    support = classification_report(y_test.iloc[:, i], y_pred[:, i], output_dict=True)['1']['support']

    metrics['Trait'].append(column)
    metrics['Accuracy'].append(accuracy)
    metrics['Precision'].append(precision)
    metrics['Recall'].append(recall)
    metrics['F1-score'].append(f1)
    metrics['Support'].append(support)

metrics_df = pd.DataFrame(metrics)

overall_accuracy = accuracy_score(y_test, y_pred)

print("\n\n",metrics_df)
print(f"\nOverall Accuracy: {overall_accuracy:.4f}")
