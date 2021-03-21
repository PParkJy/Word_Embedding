import nltk
from nltk.tokenize import word_tokenize
from glove import Glove, Corpus

# Load data
f = open("./dataset/preprocessed_data.txt", "r")
text = f.readlines()
f.close()

# Word tokenization
tokenized_words = [word_tokenize(sentence[:-1]) for sentence in text] # remove "\n" by [:-1]

corpus = Corpus()
corpus.fit(tokenized_words, window = 2)

glove = Glove(no_components=100, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=10, no_threads=5, verbose=True)
glove.add_dictionary(corpus.dictionary)

test = glove.most_similar("home")
print(test)