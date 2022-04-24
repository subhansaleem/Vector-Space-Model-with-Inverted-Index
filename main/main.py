#for punctuation
import re
#for writing dictionary to file
import json
#for tokenization
from nltk.tokenize import word_tokenize
#for lemmatization
from nltk.stem import WordNetLemmatizer, PorterStemmer
import math
import numpy as np
from pprint import pprint
from collections import Counter

lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

f = open("Stopword-List.txt", "r")
text = f.read()
stopwords = (word_tokenize(text.lower()))
vector = set()
wordDic = []

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
        # words[index] = lemmatizer.lemmatize(items)
        words[index] = ps.stem(items)

    # writing cleaned data to file
    ff = open(f"Cleaned/{i}.txt", "w")
    for word in words:
        ff.write(word + " ")

    wordDic.append(words)

    for word in words:
        vector.add(word)


# pprint(len(wordDic))
vector = list(vector)
# print(len(vector))

# dic = {
#     'term' : {
#         'idf' : value,
#         'tf': [],
#         'tf-idf': []
#     }
# }

dic = {}

for vecIndex, vecItem in enumerate(vector):
    dfCount = 0
    tempList = []

    for docIndex, docItem in enumerate(wordDic):

        # counter = Counter(docItem)
        # count = counter[vecItem]
        # tempList.append(count)
        count = docItem.count(vecItem)
        tempList.append(count)

        if count != 0:
            dfCount += 1

    dic[vecItem] = {}
    dic[vecItem]['tf'] = tempList
    dic[vecItem]['idf'] = math.log2(448/dfCount)
    dic[vecItem]['tf-idf'] = [ dic[vecItem]['idf'] * element if element != 0 else 0 for element in tempList ]

#writing dictionary to file
# json.dump(dic, open("Index.txt", 'w'))


docVector = []


temp2 = []
for i in range(448):
    temp2 = []
    for key, value in sorted(dic.items()):
        for key2,value2 in value.items():
            if key2 == 'tf-idf':
                temp = value[key2]
                temp2.append(temp[i])
                # print(i,temp[i])
    docVector.append(temp2)

docVector = np.array(docVector)
# print(docVector[0])

# v = np.array([2,5,0,3])
for i in range(448):
    docVector[i]= docVector[i] / math.sqrt(sum(docVector[i]**2))
# pprint(dic)




query = 'weak heuristic'

words = word_tokenize(query)
words = [word for word in words if word not in stopwords]


for index, item in enumerate(words):
     item = item.lower()
     # words[index] = lemmatizer.lemmatize(item)
     words[index] = ps.stem(item)

queryDic = {}

for vecIndex, vecItem in enumerate(vector):
    dfCount = 0
    tempList = []

    count = words.count(vecItem)
    tempList.append(count)

    queryDic[vecItem] = {}
    queryDic[vecItem]['tf'] = count
    queryDic[vecItem]['tf-idf'] = dic[vecItem]['idf'] * count


queryVector = []
for key, value in sorted(queryDic.items()):
    for key2, value2 in value.items():
        if key2 == 'tf-idf':
           queryVector.append(value2)

queryVector = np.array(queryVector)
queryVector = queryVector / math.sqrt(sum(queryVector**2))

# pprint(len(queryVector))
# pprint(len(docVector[0]))


for i in range(448):
    if sum(x*y for x,y in zip(docVector[i],queryVector)) > 0.0:
        print(i+1)
    # if i == 360:
    #     print(sum(x*y for x,y in zip(docVector[i],queryVector)))



























