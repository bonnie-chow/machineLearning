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
        return df[column_name].apply(lambda x: x[0][attribute_name] if len(x) > 0 else None)

    def convert_if_condition_is_met(self, df, column_name, threshold):
        dictionary = dict(df[column_name].value_counts())
        return df[column_name].apply(lambda x: 'other' if self._check_if_number_of_appearances_are_below_threshold(x, threshold, dictionary) else x)

    def extract_release_year(self, df):
        return df['release_date'].apply(lambda x: x.split('/')[2])

    def _check_if_number_of_appearances_are_below_threshold(self, name, threshold, dictionary):
        for element, number_appearances in dictionary.items():
            if element == name:
                if number_appearances < threshold:
                    return True
                return False
        return False