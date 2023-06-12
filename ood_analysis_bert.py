from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import torch
import time
import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# The vector embedding associated to each text is simply the hidden state that Bert outputs for the [CLS] token.

def ood_analysis(naug):
    print(f"Current time: {datetime.datetime.now().time()}")
    start_time = time.time()

    device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased").to(device)

    augmented_df = pd.read_csv("IMDB_500_" + str(naug) + "_ssmba_train.csv", header=None, names=["label", "text"])
    tokenized_augmented = tokenizer(augmented_df["text"].values.tolist(), padding=True, truncation=True, return_tensors="pt")
    labels_augmented = augmented_df["label"]

    original_df = pd.read_csv("IMDB_500.csv", header=None, names=["label", "text"])
    tokenized_original = tokenizer(original_df["text"].values.tolist(), padding=True, truncation=True, return_tensors="pt")
    labels_original = original_df["label"]

    tokenized_augmented = {k: torch.tensor(v).to(device) for k, v in tokenized_augmented.items()}
    tokenized_original = {k: torch.tensor(v).to(device) for k, v in tokenized_original.items()}

    with torch.no_grad():
      hidden_augmented = model(**tokenized_augmented)
      hidden_original = model(**tokenized_original)

    # Load pre-trained GloVe Word Vectors
    glove_file = 'glove.6B/glove.6B.50d.txt'
    word_vectors = {}
    with open(glove_file, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vectors = np.asarray(values[1:], dtype='float32')
            word_vectors[word] = vectors

    # Generate document vectors for the original and augmented data
    original_vectorized_data = hidden_original.last_hidden_state[:, 0, :]
    augmented_vectorized_data = hidden_augmented.last_hidden_state[:, 0, :]

    # Concatenate the original and augmented data
    all_vectorized_data = np.concatenate((original_vectorized_data, augmented_vectorized_data), axis=0)

    # Apply dimensionality reduction with PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(all_vectorized_data)

    # Separate the reduced data into original and augmented data
    reduced_original_data = reduced_data[:len(labels_original)]
    reduced_augmented_data = reduced_data[len(labels_original):]

    # Separate the reduced data based on the labels
    positive_original_data = reduced_original_data[np.array(labels_original) == 1]
    negative_original_data = reduced_original_data[np.array(labels_original) == 0]
    positive_augmented_data = reduced_augmented_data[np.array(labels_augmented) == 1]
    negative_augmented_data = reduced_augmented_data[np.array(labels_augmented) == 0]

    # Plot the reduced original and augmented data with different colors for positive and negative reviews
    plt.scatter(positive_original_data[:, 0], positive_original_data[:, 1], color='green', label='Original Positive', s=10)
    plt.scatter(negative_original_data[:, 0], negative_original_data[:, 1], color='red', label='Original Negative', s=10)
    plt.scatter(positive_augmented_data[:, 0], positive_augmented_data[:, 1], color='lightgreen', label='Augmented Positive', s=10)
    plt.scatter(negative_augmented_data[:, 0], negative_augmented_data[:, 1], color='salmon', label='Augmented Negative', s=10)
    plt.title(f"OOD generalization for 500 IMDB examples and naug={naug}")
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    # Move the legend outside the plot
    plt.legend(bbox_to_anchor=(0.5, 1.28), loc='upper center', ncol=2)

    plt.tight_layout()  # Adjust plot layout for better display
    plt.show()

    print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


if __name__ == "__main__":
    for naug in [1, 2, 4, 8, 16, 32]:
        ood_analysis(naug=naug)
