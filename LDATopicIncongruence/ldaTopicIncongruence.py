from gensim import corpora, models, utils
import numpy as np
from .data import getData
from .preprocess import preprocess, lemmatize_stemming

class LdaTopicIncongruence:
    def __init__(self, nonJokesPath, jokesPath, jokesColumnName, nonJokesColumnName, gloveModel):
        self.buildTopic2Vec(nonJokesPath, jokesPath, jokesColumnName, nonJokesColumnName, gloveModel)
        self.windowSize = 4
        self.shiftSize = 1
        self.vectorLength = 20
        print('LDA model initialization complete...')

    def buildTopic2Vec(self, nonJokesPath, jokesPath, jokesColumnName, nonJokesColumnName, gloveModel):
        print('building topic to vector model on current corpus...')
        print('getting data...')
        # import data
        nonJokesTextData = getData(nonJokesPath, nonJokesColumnName)
        jokesTextData = getData(jokesPath, jokesColumnName)
        textList = jokesTextData[jokesColumnName][:500].append(nonJokesTextData[nonJokesColumnName][:500])
        print('preprocessing data...')
        # process data
        processedDoc = textList.map(preprocess)
        # process with bag of words
        dictionary = corpora.Dictionary(processedDoc)
        # doc to bow
        bow_corpus = [dictionary.doc2bow(doc) for doc in processedDoc]
        # build LDA
        print('building LDA model on corpus...')
        self.lda_model = models.LdaMulticore(bow_corpus, num_topics=100, id2word=dictionary, passes=10, workers=2)
        # build topic vectors
        print('building topic to vector model...')
        self.topic2vec = {}
        for idx, _ in self.lda_model.print_topics(num_topics=-1):
            topicList = self.lda_model.show_topic(idx)
            topicArray = []
            probArray = []
            for (text, prob) in topicList:
                try:
                    vec = gloveModel.get_vector(text)
                    topicArray.append(np.copy(vec))
                    probArray.append(prob)
                except:
                    pass
            topicVector = np.matmul(np.reshape(np.array(probArray), (1, len(probArray))), np.array(topicArray)).reshape(25,)
            self.topic2vec[idx] = topicVector
        print('topic to vector model complete...')

    def text2vec(self, text):
        topicAnalysis = {}
        for queryData in text:
            tokenized = utils.simple_preprocess(queryData)
            start = 0
            end = start + self.windowSize
            windowContent = []
            while end <= len(tokenized):
                windowContent.append(tokenized[start:end])
                start += self.shiftSize
                end += self.shiftSize
            curDict = corpora.Dictionary(windowContent)
            curBow = [curDict.doc2bow(content) for content in windowContent]
            topicIdList = []
            for eachBow in curBow:
                top_topics = self.lda_model.get_document_topics(bow=eachBow)
                if len(top_topics) > 0:
                    topicIdList.append(self.lda_model.get_document_topics(bow=eachBow)[0][0])
                else:
                    topicIdList.append(topicIdList[-1])
            topicAnalysis[queryData] = topicIdList

        for item in topicAnalysis:
            topicList = topicAnalysis[item]
            topicAnalysis[item] = [self.topic2vec[topic] for topic in topicList]

        sentencePlot = {}
        result_vectors = []
        for item in topicAnalysis:
            vectors = topicAnalysis[item]
            diffs = [0.0]
            for i in range(1, len(vectors)):
                dist = np.linalg.norm(vectors[i]-vectors[i-1])
                diffs.append(dist)
            # if (len(diffs) > self.vectorLength):
            #     diffs = diffs[:self.vectorLength]
            # else:
            #     diffs = diffs + ([0.0] * (self.vectorLength - len(diffs)))
            sentencePlot[item] = diffs
            result_vectors.append(diffs)
        return result_vectors





