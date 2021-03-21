from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from callback import callback

# Required at first time only
# nltk.download('punkt')

# Load data
f = open("./dataset/preprocessed_data.txt", "r")
text = f.readlines()
f.close()

# Word tokenization
tokenized_words = [word_tokenize(sentence[:-1]) for sentence in text] # remove "\n" by [:-1]
#print(tokenized_words)

# Word embedding parameter
dim = 100
window_size = 2
cbow = 0
skip_gram = 1

# Training of Word2Vec
model = Word2Vec(sentences=tokenized_words, size=dim, window=window_size, sg=cbow, iter=10, compute_loss=True, callbacks=[callback()], workers=5)

# Print vector
test = model.wv["world"]
print(len(test))

# Similarity
test = model.wv.most_similar(positive=["home"])
print(test)

