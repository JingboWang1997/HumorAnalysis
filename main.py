from LDATopicIncongruence.ldaTopicIncongruence import LdaTopicIncongruence
import gensim.downloader as api

print('downloading glove model')
glove_model = api.load("glove-twitter-25")
model = LdaTopicIncongruence('./Data/abcnews-date-text.csv', './Data/shortjokes.csv', 'Joke', 'headline_text', glove_model)
print('text query')
vectors = model.text2vec([
    "The executive overseeing Facebook's recently-unveiled news section appeared to defend the company's controversial decision to include Breitbart, a far-right website known for misinformation, as one of its sources.",
    "Campbell Brown, the head of global news partnerships at Facebook, wrote in a blog post on Wednesday that she believed when 'building out a destination for news on Facebook' content 'from ideological publishers on both the left and right' should be included.",
    "Facebook unveiled its news section, which has been in the works for months, last Friday."
])

print(vectors)