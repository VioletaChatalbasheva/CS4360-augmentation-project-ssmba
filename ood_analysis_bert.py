import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
from transformers import DistilBertTokenizerFast, DistilBertModel

# Load the pre-trained DistilBERT tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')


def ood_analysis(naug, dataset, bias):
    df = pd.read_csv(dataset + '_' + bias + '_' + str(naug) + '_ssmba_train.csv', header=None, names=["label", "text"])

    augmented_reviews = df['text'][:-500]
    original_reviews = df['text'][-500:]
    augmented_labels = df['label'][:-500]
    original_labels = df['label'][-500:]

    # Tokenize and encode the text data using DistilBERT tokenizer
    encoded_data = tokenizer(list(original_reviews) + list(augmented_reviews), truncation=True, padding=True, return_tensors='pt')
    input_ids = encoded_data['input_ids']
    attention_mask = encoded_data['attention_mask']

    # Generate document embeddings using DistilBERT model
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        document_embeddings = outputs.last_hidden_state[:, 0, :].numpy()

    # Separate the document embeddings into original and augmented data
    original_embeddings = document_embeddings[-500:]
    augmented_embeddings = document_embeddings[:-500]

    # Apply dimensionality reduction with PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(np.concatenate((original_embeddings, augmented_embeddings), axis=0))

    # Separate the reduced data into original and augmented data
    reduced_original_data = reduced_data[:500]
    reduced_augmented_data = reduced_data[500:]

    # Separate the reduced data based on the labels
    positive_original_data = reduced_original_data[np.array(original_labels) == 1]
    negative_original_data = reduced_original_data[np.array(original_labels) == 0]
    positive_augmented_data = reduced_augmented_data[np.array(augmented_labels) == 1]
    negative_augmented_data = reduced_augmented_data[np.array(augmented_labels) == 0]

    # Plot the reduced original and augmented data with different colors for positive and negative reviews
    plt.scatter(positive_original_data[:, 0], positive_original_data[:, 1], color='green', label='Original Positive', s=10)
    plt.scatter(negative_original_data[:, 0], negative_original_data[:, 1], color='red', label='Original Negative', s=10)
    plt.scatter(positive_augmented_data[:, 0], positive_augmented_data[:, 1], color='lightgreen', label='Augmented Positive', s=10)
    plt.scatter(negative_augmented_data[:, 0], negative_augmented_data[:, 1], color='salmon', label='Augmented Negative', s=10)
    if bias == "bias":
        plt.title(f"OOD generalization on {dataset} biased examples with naug={naug}")
    else:
        plt.title(f"OOD generalization on {dataset} naive examples with naug={naug}")
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    # Move the legend outside the plot
    plt.legend(bbox_to_anchor=(0.5, 1.28), loc='upper center', ncol=2)
    plt.savefig(f'../results/ood/{dataset}_{bias}_{naug}', bbox_inches='tight')

    plt.tight_layout()  # Adjust plot layout for better display
    plt.show()


if __name__ == "__main__":
    for dataset in ["IMDB", "MNLI"]:
        for bias in ["no_bias", "bias"]:
          # , 2, 4, 8, 16, 32
            for naug in [1]:
                ood_analysis(naug=naug, dataset=dataset, bias=bias)
