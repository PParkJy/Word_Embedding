from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from callback import callback
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt


# Load data
f = open("./dataset/preprocessed_data.txt", "r")
text = f.readlines()
f.close()

# Word tokenization
tokenized_words = [word_tokenize(sentence[:-1]) for sentence in text] # remove "\n" by [:-1]

# Word embedding parameter
dim = 100
window_size = 2
cbow = 0
skip_gram = 1

# Training of Word2Vec
model = Word2Vec(sentences=tokenized_words, size=dim, window=window_size, sg=cbow, iter=10, compute_loss=True, callbacks=[callback()], workers=5)

# Print vector
test = model.wv["world"]
#print(test)) 

# Cosine similarity
test = model.wv.most_similar(positive=["happy"]) # negative option
#print(test)

# Check the distribution by TSNE
vocab = list(model.wv.vocab)
X = model[vocab]
#print(len(X))

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X[:200,:]) # select 200 words

df = pd.DataFrame(X_tsne, index=vocab[:200], columns=['x', 'y'])
#print(df.head(10))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(df['x'], df['y'])

for word, pos in df.iterrows():
    ax.annotate(word, pos, fontsize=10)

plt.show()
