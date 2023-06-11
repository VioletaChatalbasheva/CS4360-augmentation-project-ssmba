import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load pre-trained GloVe Word Vectors
glove_file = 'glove.6B/glove.6B.50d.txt'
word_vectors = {}
with open(glove_file, encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vectors = np.asarray(values[1:], dtype='float32')
        word_vectors[word] = vectors

# A list of original data and augmented data with their corresponding labels
original_reviews = ['The movie was great and highly enjoyable.',
                    'I found the film to be disappointing.']
augmented_reviews = ['The movie was fantastic and extremely enjoyable.',
                     'I was disappointed by the film.']
labels = [1, 0]


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
positive_augmented_data = reduced_augmented_data[np.array(labels) == 1]
negative_augmented_data = reduced_augmented_data[np.array(labels) == 0]

# Plot the reduced original and augmented data with different colors for positive and negative reviews
plt.scatter(positive_original_data[:, 0], positive_original_data[:, 1], color='green', label='Original Positive')
plt.scatter(negative_original_data[:, 0], negative_original_data[:, 1], color='red', label='Original Negative')
plt.scatter(positive_augmented_data[:, 0], positive_augmented_data[:, 1], color='lightgreen', label='Augmented Positive')
plt.scatter(negative_augmented_data[:, 0], negative_augmented_data[:, 1], color='salmon', label='Augmented Negative')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Move the legend outside the plot
plt.legend(bbox_to_anchor=(0.5, 1.20), loc='upper center', ncol=2)

plt.tight_layout()  # Adjust plot layout for better display
plt.show()
