import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load data
margin_features = pd.read_csv(r'C:\Users\Shrey\Documents\Margin-Detection\features_auto_81.csv').iloc[:29]
personality_traits = pd.read_csv(r'C:\Users\Shrey\Documents\Margin-Detection\labels_traits_29.csv')

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

# Metrics calculation
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

print("\n\n", metrics_df)
print(f"\nOverall Accuracy: {overall_accuracy:.4f}")

# Function to predict personality traits based on new margin features
def predict_personality(new_data):
    if isinstance(new_data, dict):
        new_data = pd.DataFrame([new_data])  # Convert dict to DataFrame
    elif isinstance(new_data, list):
        new_data = pd.DataFrame(new_data, columns=margin_features.columns)  # Convert list to DataFrame
    
    predictions = multi_output_model.predict(new_data)
    prediction_df = pd.DataFrame(predictions, columns=personality_traits.columns)
    
    return prediction_df

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

predicted_traits = predict_personality(new_margin_data)
print("\nPredicted Personality Traits:\n", predicted_traits)
