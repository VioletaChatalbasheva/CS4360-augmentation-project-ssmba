import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def ood_analysis(naug):
    # Load pre-trained GloVe Word Vectors
    glove_file = 'glove.6B/glove.6B.50d.txt'
    word_vectors = {}
    with open(glove_file, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vectors = np.asarray(values[1:], dtype='float32')
            word_vectors[word] = vectors

    with open('test/input.txt', "r", encoding="utf-8") as input_file, open('test/labels.txt', "r") as labels_file, \
            open('ssmba_out_imdb_500_' + str(naug), "r", encoding="utf-8") as augmented_input_file, open('ssmba_out_imdb_500_' + str(naug) + '.label', "r") as augmented_labels_file:
        original_reviews = input_file.readlines()

        labels = []
        for label in labels_file.readlines():
            labels.append(int(label))

        augmented_reviews = []
        for augmented_review in augmented_input_file.readlines():
            augmented_reviews.append(augmented_review.rstrip())

        augmented_labels = []
        for label in augmented_labels_file.readlines():
            augmented_labels.append(int(label))

    # Generate document vectors by averaging word vectors
    def generate_document_vector(text):
        vectors = [word_vectors[word] for word in text.split() if word in word_vectors]
        if not vectors:
            return np.zeros(50)  # Use the appropriate vector size based on the downloaded GloVe vectors
        return np.mean(vectors, axis=0)

    # Generate document vectors for the original and augmented data
    original_vectorized_data = np.array([generate_document_vector(review) for review in original_reviews])
    augmented_vectorized_data = np.array([generate_document_vector(review) for review in augmented_reviews])

    # Concatenate the original and augmented data
    all_vectorized_data = np.concatenate((original_vectorized_data, augmented_vectorized_data), axis=0)

    # Apply dimensionality reduction with PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(all_vectorized_data)

    # Separate the reduced data into original and augmented data
    reduced_original_data = reduced_data[:len(original_reviews)]
    reduced_augmented_data = reduced_data[len(original_reviews):]

    # Separate the reduced data based on the labels
    positive_original_data = reduced_original_data[np.array(labels) == 1]
    negative_original_data = reduced_original_data[np.array(labels) == 0]
    positive_augmented_data = reduced_augmented_data[np.array(augmented_labels) == 1]
    negative_augmented_data = reduced_augmented_data[np.array(augmented_labels) == 0]

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


if __name__ == "__main__":
    for naug in [1, 2, 4, 8, 16, 32]:
        ood_analysis(naug=naug)
