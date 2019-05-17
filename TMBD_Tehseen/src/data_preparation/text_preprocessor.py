import warnings

import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from src.data_preparation.column_selector import ColumnSelector


class TextPreprocessor:

    def create_embedding_for(self, word_feature, cut_rare_words):
        warnings.filterwarnings('ignore')

        EMBED_SIZE = 50
        MAX_FEATURES = 3000
        MAX_LEN = 20

        column_selector = ColumnSelector()

        train = pd.read_csv("../resources/train.csv")
        column = column_selector.consider_subset_of_columns(train, [word_feature]).astype(str).values
        feature = column.reshape(3000,)
        print(feature.shape)

        tokenizer = Tokenizer(num_words=MAX_FEATURES)
        tokenizer.fit_on_texts(list(feature))

        feature = tokenizer.texts_to_sequences(feature)

        word_index = tokenizer.word_index
        print('# of unique tokens: ', len(word_index))
        print(list(tokenizer.index_word.items())[0:10])
        print(list(tokenizer.index_word.items())[-10:])

        for item in feature:
            for index in range(0, 20):
                while item.count(index) > 0:
                    if index in item:
                        item.remove(index)

        if cut_rare_words is True:
            for item in feature:
                for index in range(18000, 18539):
                    while item.count(index) > 0:
                        if index in item:
                            item.remove(index)

        feature = pad_sequences(feature, maxlen=MAX_LEN)

        print(feature)

        embeddings_index = {}
        f = open('glove.6B.50d.txt', encoding='utf8')
        for line in f:
           values = line.split()
           word = values[0]
           coefs = np.asarray(values[1:], dtype='float32')
           embeddings_index[word] = coefs
        f.close()

        print('Total %s word vectors in Glove 6B 100d.' % len(embeddings_index))

        embedding_matrix = np.random.random((len(word_index) + 1, EMBED_SIZE))
        for word, i in word_index.items():
           embedding_vector = embeddings_index.get(word)
           if embedding_vector is not None:
               embedding_matrix[i] = embedding_vector

        return embedding_matrix
