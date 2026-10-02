import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from score_traits import predict as rule_predict

HERE = os.path.dirname(os.path.abspath(__file__))

# Load data
margin_features = pd.read_csv(os.path.join(HERE, 'features_auto_81.csv')).iloc[:29]
personality_traits = pd.read_csv(os.path.join(HERE, 'labels_traits_29.csv'))

# Reset index
margin_features.reset_index(drop=True, inplace=True)
personality_traits.reset_index(drop=True, inplace=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(margin_features, personality_traits, test_size=0.25, random_state=42)

# Model initialization
nb_model = BernoulliNB()
multi_output_model = MultiOutputClassifier(nb_model)

# Model training
multi_output_model.fit(X_train, y_train)

# Predictions on test data
y_pred = multi_output_model.predict(X_test)

# Rule-based predictions from the margin/trait weight matrix.
rule_pred = rule_predict(X_test)

# Metrics calculation
metrics = {
    'Trait': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1-score': [],
    'Support': [],
    'Rule Accuracy': []
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
    # Label columns are in score_traits.TRAITS order, so match by position.
    metrics['Rule Accuracy'].append(accuracy_score(y_test.iloc[:, i], rule_pred.iloc[:, i]))

metrics_df = pd.DataFrame(metrics)

overall_accuracy = accuracy_score(y_test, y_pred)

print("\n\n", metrics_df)
print(f"\nOverall Accuracy: {overall_accuracy:.4f}")


def rule_accuracy(margins, labels):
    """Mean per-trait accuracy of the weight matrix, against the base rate."""
    pred = rule_predict(margins)
    acc = [(pred.iloc[:, i].values == labels.iloc[:, i].values).mean()
           for i in range(labels.shape[1])]
    base = [max(labels.iloc[:, i].mean(), 1 - labels.iloc[:, i].mean())
            for i in range(labels.shape[1])]
    return np.mean(acc), np.mean(base)


# Scored over all 29 labelled papers rather than the 8-row test split, which is
# too small to separate anything. Expert margins are the ceiling the weights can
# reach; the detector's own margins are what the pipeline achieves today.
expert_margins = pd.read_csv(os.path.join(HERE, 'features_manual_29.csv'))
expert_margins.columns = [c.upper() for c in expert_margins.columns]
expert_margins = expert_margins.rename(columns={'RLM': 'RFLM'})

for name, margins in [('expert margins  ', expert_margins),
                      ('detector margins', margin_features)]:
    acc, base = rule_accuracy(margins, personality_traits)
    print(f"weight matrix on {name}: {acc * 100:.1f}%   (base rate {base * 100:.1f}%)")

# Function to predict personality traits based on new margin features
def predict_personality(new_data):
    """Returns (naive-Bayes predictions, weight-matrix predictions)."""
    if isinstance(new_data, dict):
        new_data = pd.DataFrame([new_data])  # Convert dict to DataFrame
    elif isinstance(new_data, list):
        new_data = pd.DataFrame(new_data, columns=margin_features.columns)  # Convert list to DataFrame
    
    nb_df = pd.DataFrame(multi_output_model.predict(new_data),
                         columns=personality_traits.columns)
    rule_df = rule_predict(new_data).set_axis(personality_traits.columns, axis=1)

    return nb_df, rule_df

# Example usage
new_margin_data = {

    'SLM':0,
    'WLM': 0,
    'DAFLM': 1,
    'RFLM': 0,
    'CCLM': 0,
    'CVLM':0,
    'TT': 1,
    'TS':0,
    'TDA':0,
    'BT':0,
    'BS':0,
    'BDA':1
}

nb_traits, rule_traits = predict_personality(new_margin_data)
print("\nPredicted Personality Traits (naive Bayes):\n", nb_traits)
print("\nPredicted Personality Traits (weight matrix):\n", rule_traits)
