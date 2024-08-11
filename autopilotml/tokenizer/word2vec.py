import gensim
from gensim.models import Word2Vec
import gensim.downloader as api


class Word2VecTokenizer:
    def __init__(self):
        self.model = api.load("word2vec-google-news-300")

    def tokenize(self, text):
        return self.model[text]

    def __call__(self, text):
        return self.tokenize(text)
    
    def similar(self, word):
        return self.model.most_similar(word)
    
    def finetune(self, data):
        self.model.build_vocab(data, update=True)
        self.model.train(data, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        return self.model
