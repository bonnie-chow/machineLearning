
class ColumnDataPreprocessor:

    def preprocess_languages(self, df):
        df['original_language'].apply(lambda x: 'other' if x != 'en' else 'en')
