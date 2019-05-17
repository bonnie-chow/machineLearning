from src.data_preparation.text_preprocessor import TextPreprocessor


class ColumnDataPreprocessor:

    text_preprocessor = TextPreprocessor()

    def preprocess_languages(self, df):
        df['original_language'].apply(lambda x: 'other' if x != 'en' else 'en')

    def map_embeddings_df_columns(self, df, feature1, feature2):
        tagline_embeddings = self.create_embedding_array(df, feature1, False)
        overview_embeddings = self.create_embedding_array(df, feature2, True)
        return tagline_embeddings, overview_embeddings

    def create_embedding_array(self, df, feature, cut_rare_words):
        embedding = self.text_preprocessor.create_embedding_for(feature, cut_rare_words)
        embeddings = []
        for index, row in df.iterrows():
            row[feature] = embedding[index]
            embeddings.append(row[feature])
        return embeddings