import ast
import pandas as pd


class DictionaryDataExtractor:

    dict_columns = ['belongs_to_collection', 'genres', 'production_companies',
                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']

    def text_to_dict(self, df, dict_columns=dict_columns):
        for column in dict_columns:
            df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x))
        return df

    def extract_data_from_dict(self, df, column_name, attribute_name):
        return df[column_name].apply(lambda x: x[0][attribute_name] if x != {} else 0)
