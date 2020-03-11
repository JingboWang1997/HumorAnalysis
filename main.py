from LDATopicIncongruence.ldaTopicIncongruence import LdaTopicIncongruence
from LDATopicIncongruence.data import getData
import gensim.downloader as api
import random

print('downloading glove model...')
glove_model = api.load("glove-twitter-25")
model = LdaTopicIncongruence('./Data/abcnews-date-text.csv', './Data/shortjokes.csv', 'Joke', 'headline_text', glove_model)

print('testing...')
jokeCount = 500
nonJokeCount = 500
nonJokesTextData = getData('./Data/abcnews-date-text.csv', 'headline_text')
jokesTextData = getData('./Data/shortjokes.csv', 'Joke')
textList = jokesTextData['Joke'][:jokeCount].append(nonJokesTextData['headline_text'][:nonJokeCount])
vectors = model.text2vec(textList)

bestThreshold = 0.7
bestAccuracy = 0
counter = 0
while True:
    threshold = bestThreshold + random.uniform(-0.5, 0.5)
    if threshold > 0 and threshold < 1:
        jokeCorrect = 0
        jokeError = 0
        nonJokeCorrect = 0
        nonJokeError = 0
        for i, v in enumerate(vectors):
            maxTopicChange = max(v)
            if i < jokeCount:
                if maxTopicChange > threshold:
                    jokeCorrect += 1
                else:
                    jokeError += 1
            else:
                if maxTopicChange > threshold:
                    nonJokeError += 1
                else:
                    nonJokeCorrect + 1
        accuracy = (jokeCorrect + nonJokeCorrect) / (jokeCorrect + jokeError + nonJokeCorrect + nonJokeError)
        if (accuracy > bestAccuracy):
            bestThreshold = threshold
            bestAccuracy = accuracy
        # print('threshold: ', threshold)
        # print('accuracy: ', accuracy)
        counter += 1
        if counter % 10 == 0:
            print('bestThreshold: ', bestThreshold)
            print('bestAccuracy: ', bestAccuracy)
            counter = 0