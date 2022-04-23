#for punctuation
import re
#for writing dictionary to file
import json
#for tokenization
from nltk.tokenize import word_tokenize
#for lemmatization
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

f = open("Stopword-List.txt", "r")
text = f.read()
stopwords = (word_tokenize(text.lower()))
vector = set()

# looping over all the files
for i in range(1, 449):
    # reading from file
    f = open(f"Abstracts/{i}.txt", "r")
    text = f.read().lower()

    # removing punctuation
    text = re.sub(r'[^\w\s]', ' ', text)

    # tokenization  with nltk
    words = word_tokenize(text)

    # removing stopwords
    words = [word for word in words if word not in stopwords]

    # lemmetization  with nltk.stem - WordNetLemmatizer

    for index, items in enumerate(words):
        words[index] = lemmatizer.lemmatize(items)

    # writing cleaned data to file
    # ff = open(f"Cleaned/{i}.txt", "w")
    # for word in words:
    #     ff.write(word + " ")

    for word in words:
        vector.add(word)


print(len(vector))
