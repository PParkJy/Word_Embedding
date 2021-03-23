import nltk
from nltk.tokenize import word_tokenize
from glove import Glove, Corpus
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load data
f = open("./dataset/preprocessed_data.txt", "r")
text = f.readlines()
f.close()

# Word tokenization
tokenized_words = [word_tokenize(sentence[:-1]) for sentence in text] # remove "\n" by [:-1]

corpus = Corpus()
corpus.fit(tokenized_words, window = 2)

# Training of GloVe
glove = Glove(no_components=100, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=10, no_threads=5, verbose=True)
glove.add_dictionary(corpus.dictionary) #?

# Print vector
test = glove.word_vectors[glove.dictionary["world"]]
#print(test)

# Cosine similarity
test = glove.most_similar("happy")
#print(test)

'''
# Check the distribution by TSNE
vocab = list(glove.dictionary.keys())
vectors = [glove.dictionary[word] for word in vocab]
vectors = np.reshape(vectors, (-1,1))

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(vectors[:200, :])

df = pd.DataFrame(X_tsne, index=vocab[:200], columns=['x', 'y'])
#print(df.head(10))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(df['x'], df['y'])

for word, pos in df.iterrows():
    ax.annotate(word, pos, fontsize=10)

plt.savefig("glove.png")
'''