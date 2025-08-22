import os
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack

# Function to recursively load JSON files from a directory and extract both API calls and time, with sorting by time
def load_data_from_directory(directory):
    api_calls = []
    labels = []
    times = []  # Store the "time" properties

    # Traverse the directory and subdirectories recursively
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".json"):  # Only process .json files
                file_path = os.path.join(root, filename)
                with open(file_path, 'r') as f:
                    try:
                        # Load the JSON data from file
                        data = json.load(f)
                        # Debugging: Check the data type and structure
                        if not isinstance(data, list):
                            print(f"Error: Data is not a list! Data type: {type(data)} in file: {file_path}")
                            continue
                        # Sort the entries based on the "time" property (Unix timestamp)
                        data_sorted = sorted(data, key=lambda x: x['time'])
                        
                        # Extract sorted API call names (bigrams/trigrams)
                        api_call_sequence = ' '.join([entry['api'] for entry in data_sorted])
                        api_calls.append(api_call_sequence)
                        
                        # Convert Unix timestamps to a more usable format (optional)
                        time_values = [entry['time'] for entry in data_sorted]
                        times.append(np.mean(time_values))  # Use the average time for simplicity

                        # Label: 1 for malware, 0 for benign (based on directory name)
                        if 'malware' in root:
                            labels.append(1)
                        else:
                            labels.append(0)
                    except json.JSONDecodeError:
                        continue

    return api_calls, labels, times

# Directories for training and testing data
train_PE_benign_dir = "[static path for data files]/vkl-benign/PE-format/train"
train_know_malware_dir = "[static path for data files]/vkl-malware/known/train"
test_know_malware_dir = "[static path for data files]/vkl-malware/known/test"
test_unknow_malware_dir = "[static path for data files]/vkl-malware/unknown/test"
test_unknow_clam_malware_dir = "[static path for data files]/vkl-malware/unknown-clamav/test"
test_PE_benign_dir = "[static path for data files]/vkl-benign/PE-format/test"
test_PE_extension_benign_dir = "[static path for data files]/vkl-benign/PE-extension/test"
train_PE_extension_benign_dir = "[static path for data files]/vkl-benign/PE-extension/train"

# Load data recursively from directories
train_know_malware_api_calls, train_know_malware_labels, train_know_malware_times = load_data_from_directory(train_know_malware_dir)
train_PE_benign_api_calls, train_PE_benign_labels, train_PE_benign_times = load_data_from_directory(train_PE_benign_dir)
train_PE_extension_benign_api_calls, train_PE_extension_benign_labels, train_PE_extension_benign_times = load_data_from_directory(train_PE_extension_benign_dir)
# test_know_malware_api_calls, test_know_malware_labels, test_know_malware_times = load_data_from_directory(test_know_malware_dir)
# test_unknow_malware_api_calls, test_unknow_malware_labels, test_unknow_malware_times = load_data_from_directory(test_unknow_malware_dir)
test_unknow_clam_malware_api_calls, test_unknow_clam_malware_labels, test_unknow_clam_malware_times = load_data_from_directory(test_unknow_clam_malware_dir)
test_PE_benign_api_calls, test_PE_benign_labels, test_PE_benign_times = load_data_from_directory(test_PE_benign_dir)
test_PE_extension_api_calls, test_PE_extension_labels, test_PE_extension_times = load_data_from_directory(test_PE_extension_benign_dir)

# Combine training data
X_train = train_know_malware_api_calls + train_PE_benign_api_calls + train_PE_extension_benign_api_calls
y_train = train_know_malware_labels + train_PE_benign_labels + train_PE_extension_benign_labels
times_train = train_know_malware_times + train_PE_benign_times + train_PE_extension_benign_times

# Combine test data
X_test = test_unknow_clam_malware_api_calls + test_PE_benign_api_calls + test_PE_extension_api_calls
y_test = test_unknow_clam_malware_labels + test_PE_benign_labels + test_PE_extension_labels
times_test = test_unknow_clam_malware_times + test_PE_benign_times + test_PE_extension_times

X_train = [doc for doc in X_train if doc.strip()]  # Remove empty or whitespace-only documents
X_test = [doc for doc in X_test if doc.strip()]
# Vectorize the API calls using Trigrams model (CountVectorizer)
vectorizer = CountVectorizer(ngram_range=(3, 3))  # Trigrams
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Normalize the time features using MinMaxScaler
scaler = MinMaxScaler()
times_train_scaled = scaler.fit_transform(np.array(times_train).reshape(-1, 1))
times_test_scaled = scaler.transform(np.array(times_test).reshape(-1, 1))

# Convert the time data into a sparse matrix to concatenate it with the vectorized data
X_train_final = hstack([X_train_vec, times_train_scaled])
X_test_final = hstack([X_test_vec, times_test_scaled])

# Initialize and train Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_final, y_train)

# Evaluate the model on the test set
y_pred = clf.predict(X_test_final)

# Print evaluation results: Precision, Recall, F1-Score
# print("Classification Report:")
# print(classification_report(y_test, y_pred, target_names=['Benign', 'Malware']))
report = classification_report(y_test, y_pred, target_names=['Benign', 'Unknown ClamAV Malware'], output_dict=True)

# Print the formatted classification report with percentages
print("Classification Report (in percentages):")
for label, metrics in report.items():
    if label not in ['accuracy', 'macro avg', 'weighted avg']:  # Skip non-class rows
        print(f"{label}:")
        print(f"  Precision: {metrics['precision'] * 100:.2f}%")
        print(f"  Recall: {metrics['recall'] * 100:.2f}%")
        print(f"  F1-Score: {metrics['f1-score'] * 100:.2f}%")
    else:
        # Print accuracy, macro avg, and weighted avg in percentages as well
        if label == 'accuracy':
            print(f"Accuracy: {metrics * 100:.2f}%")
        else:
            print(f"{label}:")
            print(f"  Precision: {metrics['precision'] * 100:.2f}%")
            print(f"  Recall: {metrics['recall'] * 100:.2f}%")
            print(f"  F1-Score: {metrics['f1-score'] * 100:.2f}%")
# Sample print of number of training and test samples
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

