import re
from lxml import etree
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

targetXML = open('./dataset/ted_en-20160408.xml', 'r', encoding='utf-8')
target_text = etree.parse(targetXML)
parse_text = '\n'.join(target_text.xpath('//content/text()'))

content_text = re.sub(r'\([^)]*\)', '', parse_text) 

sent_text = sent_tokenize(content_text) # sentence tokenization

# remove ".", Upper case -> Lower case
normalized_text = []
for string in sent_text:
     tokens = re.sub(r"[^a-z0-9]+", " ", string.lower())
     normalized_text.append(tokens)

f = open('./dataset/preprocessed_data.txt', "w")
for i in normalized_text:
    f.write(i + "\n")
f.close()