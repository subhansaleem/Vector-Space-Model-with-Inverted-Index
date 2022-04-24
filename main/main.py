#for punctuation
import re
#for writing dictionary to file
import json
#for tokenization
from nltk.tokenize import word_tokenize
#for lemmatization
from nltk.stem import WordNetLemmatizer, PorterStemmer
#for sqrt func
import math
#for numpy array
import numpy as np

lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

# creating stopword list
def stopwds():
    f = open("Stopword-List.txt", "r")
    text = f.read()
    ls = (word_tokenize(text.lower()))
    return ls

# for creating index with words, tf, idf, and tf-idf values

    # dic = {
    #     'term' : {
    #         'idf' : value,
    #         'tf': [],
    #         'tf-idf': []
    #     }
    # }

def createIndex():
    wordDic = []
    # for all words in 448 documents
    vector = set()

    # stopwords = (word_tokenize(text.lower()))
    stopwords = stopwds()

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

        # lemmetization or stemming with nltk.stem - WordNetLemmatizer, PorterStemmer

        for index, items in enumerate(words):
            #lemmetizating
            # words[index] = lemmatizer.lemmatize(items)

            #stemming
            words[index] = ps.stem(items)

        # writing cleaned data to file
        ff = open(f"Cleaned/{i}.txt", "w")
        for word in words:
            ff.write(word + " ")

        # creating word list of documents
        wordDic.append(words)

        for word in words:
            vector.add(word)

    vector = list(vector)
    dic = {}

    for vecIndex, vecItem in enumerate(vector):
        #df value
        dfCount = 0
        tempList = []

        for docIndex, docItem in enumerate(wordDic):
            count = docItem.count(vecItem)
            tempList.append(count)

            if count != 0:
                dfCount += 1

        #creating dic for index
        dic[vecItem] = {}
        dic[vecItem]['tf'] = tempList
        dic[vecItem]['idf'] = math.log2(448/dfCount)
        dic[vecItem]['tf-idf'] = [ dic[vecItem]['idf'] * element if element != 0 else 0 for element in tempList ]

    #writing dictionary to file
    json.dump(dic, open("Index.txt", 'w'))

    #writing vector to file
    json.dump(vector, open("vector.txt",'w'))

#searching query in documents
def search(query):
    alpha = 0.001
    #loading index file
    dic = json.load(open("Index.txt"))
    #loading vector file
    vector = json.load(open("vector.txt"))
    stopwords = stopwds()
    docVector = []

    #creating a vector per doc for calculation
    for i in range(448):
        temp2 = []
        for key, value in sorted(dic.items()):
            for key2,value2 in value.items():
                if key2 == 'tf-idf':
                    temp = value[key2]
                    temp2.append(temp[i])
        docVector.append(temp2)

    #converting to numpy array for calculation
    docVector = np.array(docVector)

    #normalizing vector for optimization
    for i in range(448):
        docVector[i]= docVector[i] / math.sqrt(sum(docVector[i]**2))

    #tokening query
    words = word_tokenize(query)
    #removing stopwords
    words = [word for word in words if word not in stopwords]


    for index, item in enumerate(words):
         item = item.lower()

         #lemmatizing
         # words[index] = lemmatizer.lemmatize(item)

         #stemming
         words[index] = ps.stem(item)

    queryDic = {}

    #creating tf,tf-idf values for query
    for vecIndex, vecItem in enumerate(vector):
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

    #normalizing query for optimization
    queryVector = queryVector / math.sqrt(sum(queryVector**2))

    #calculating for product of documents and query
    for i in range(448):
        ans = sum(x * y for x, y in zip(docVector[i], queryVector))
        if  ans > alpha:
            print(i+1)

def main():
    # for creating index
    # createIndex()

    while True:
        print("\n\n1. Search Query\n2. Exit")
        inp = input()
        if int(inp) == 1:
            query = input("Enter Query: ")
            search(query)
        else:
            exit()

# drive code
if __name__ == "__main__":
    main()


















