from keras import Sequential, Input
from keras.layers import Dense, Dropout, Embedding, GlobalAveragePooling1D, Concatenate, Conv1D, MaxPooling1D

from src.data_preparation.text_preprocessor import TextPreprocessor


class MixedModel:

    textPreprocessor = TextPreprocessor()

    def build_model(self):
        model = Sequential()

        model.add(Dense(1024, input_shape=(109,)))

        model.add(Dense(512, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='relu'))

        embedded_sequences = self.textPreprocessor.create_embedding_for('tagline', False)

        x = Conv1D(128, 2, activation='relu')(embedded_sequences)
        x = GlobalAveragePooling1D()(x)
        x = Dense(1, activation='relu')(x)


        merged = Concatenate([model, x])
        return merged
