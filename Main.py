# By adding the related libraries and running the program it is going to print f1-score and standard deviation
# The plot will also be loaded, it indicates the Precision-Recall Curve with Average Precision score
import numpy as np
import pandas as pd
import matplotlib
from skbio import DNA
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  f1_score
from sklearn.metrics import precision_recall_curve, average_precision_score

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

dataset = pd.read_csv("Data.tsv", header=None, sep="\t")

dna_strings = dataset.iloc[:, 0]
classes = dataset.iloc[:, -1]

# Define a dictionary to map nucleotides to indices
nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def kmer_frequencies_with_padding(dna_strings, k):

    frequencies = []
    max_seq_length = max(len(dna) for dna in dna_strings)

    for dna_string in dna_strings:
        seq = DNA(dna_string)
        freq_dict = seq.kmer_frequencies(k=k, relative=True)

        # Convert dictionary to list of frequencies
        freq_list = [freq_dict.get(kmer, 0) for kmer in sorted(freq_dict)]

        # Pad the frequency list if necessary
        if len(freq_list) < max_seq_length:
            pad_length = max_seq_length - len(freq_list)
            freq_list.extend([0] * pad_length)

        # Append padded frequency list to the main list
        frequencies.append(freq_list)

    return np.array(frequencies)


def one_hot_encode_sequences(dna_strings):
    num_samples = len(dna_strings)
    num_features = 4  # A, C, G, T
    max_sequence_length = max(len(dna) for dna in dna_strings)

    encoded_sequences = np.zeros((num_samples, max_sequence_length * num_features), dtype=np.int8)

    for i, dna_string in enumerate(dna_strings):
        for j, nucleotide in enumerate(dna_string):
            if j < max_sequence_length:  # Only encode up to max_sequence_length
                if nucleotide in nucleotide_map:
                    index = nucleotide_map[nucleotide]
                    encoded_sequences[i, j * num_features + index] = 1
            else:
                break  # Stop encoding when sequence length exceeds max_sequence_length

        # Pad with zeros if sequence is shorter than max_sequence_length
        pad_length = max_sequence_length - len(dna_string)
        if pad_length > 0:
            encoded_sequences[i, (len(dna_string) * num_features):] = np.zeros(pad_length * num_features, dtype=np.int8)

    return encoded_sequences

# First encoding by using di-nucleotide k=2
k = 2
diNcleotide_kmer_encoded = kmer_frequencies_with_padding(dna_strings, k)

# Second encoding by using tri-nucleotides k=3
k = 3
triNucleotides_kmer_encoded = kmer_frequencies_with_padding(dna_strings, k)

# Third encoding method by using one hot encoding
encoded_sequences = one_hot_encode_sequences(dna_strings)

def stratified_k_fold_rf(X, y):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    f1_scores = []

    params = {
        'n_estimators': 100,
        'max_depth': 2,
        'class_weight': 'balanced'
    }

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        rf = RandomForestClassifier(**params, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        f1 = f1_score(y_test, y_pred)
        f1_scores.append(f1)

        mean_f1_score = np.mean(f1_scores)
        std_f1_score = np.std(f1_scores)
    return rf, mean_f1_score, std_f1_score


rf_diNcleotide, mean_f1_diNcleotide, std_f1_diNcleotide  = stratified_k_fold_rf(diNcleotide_kmer_encoded, classes)
rf_triNucleotides, mean_f1_triNucleotides, std_f1_triNucleotides = stratified_k_fold_rf(triNucleotides_kmer_encoded, classes)
rf_one_hot, mean_f1_one_hot, std_f1_one_hot = stratified_k_fold_rf(encoded_sequences, classes)

print("Mean F1 Score (diNcleotide):", mean_f1_diNcleotide)
print("Standard Deviation of F1 Scores (diNcleotide):", std_f1_diNcleotide, "\n")

print("Mean F1 Score (triNucleotides):", mean_f1_triNucleotides)
print("Standard Deviation of F1 Scores (triNucleotides):", std_f1_triNucleotides, "\n")

print("Mean F1 Score (One-Hot):", mean_f1_one_hot)
print("Standard Deviation of F1 Scores (One-Hot):", std_f1_one_hot, "\n")

# Plot Precision-Recall for the encoded methods

def plot_precision_recall_curve(y_true, y_scores, label):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    average_precision = average_precision_score(y_true, y_scores)
    plt.plot(recall, precision, label=f'{label} (AP = {average_precision:.2f})')

plt.figure(figsize=(8, 6))

# Plot precision-recall curves for each model
plot_precision_recall_curve(classes, rf_diNcleotide.predict_proba(diNcleotide_kmer_encoded)[:, 1], label='diNcleotide')
plot_precision_recall_curve(classes, rf_triNucleotides.predict_proba(triNucleotides_kmer_encoded)[:, 1], label='triNucleotides')
plot_precision_recall_curve(classes, rf_one_hot.predict_proba(encoded_sequences)[:, 1], label='One-Hot')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid(True)
plt.show()