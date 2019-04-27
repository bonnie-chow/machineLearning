import warnings

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from src.column_data_preprocessor import ColumnDataPreprocessor
from src.column_selector import ColumnSelector
from src.data_frame_selector import DataFrameSelector
from src.dictionary_data_extractor import DictionaryDataExtractor

warnings.filterwarnings('ignore')

dictionary_data_extractor = DictionaryDataExtractor()
column_selector = ColumnSelector()
column_data_preprocessor = ColumnDataPreprocessor()

train = pd.read_csv("../resources/train.csv")
test = pd.read_csv("../resources/test.csv")

train = dictionary_data_extractor.text_to_dict(train)
test = dictionary_data_extractor.text_to_dict(test)

train_without_id = column_selector.remove_unwanted_columns(train, ['id'])

reduced_train_set = column_selector.consider_subset_of_columns(train_without_id, ['budget', 'genres', 'original_language', 'popularity', 'runtime', 'revenue'])


reduced_train_set['cleansed_languages'] = str(column_data_preprocessor.preprocess_languages(reduced_train_set))
print(reduced_train_set['cleansed_languages'].value_counts())

reduced_train_set['main_genre'] = str(dictionary_data_extractor.extract_data_from_dict(reduced_train_set, 'genres', 'name'))
print(reduced_train_set.head())

train_after_handling_dicts = column_selector.remove_unwanted_columns(reduced_train_set, ['original_language', 'genres'])
print(train_after_handling_dicts.info())


train_labels = train_after_handling_dicts['revenue']

cat_attributes = ['cleansed_languages', 'main_genre']
num_attributes = ['budget', 'popularity', 'runtime']

num_pipeline = Pipeline([
    ('select_numeric', DataFrameSelector(num_attributes)),
    ('impute_age', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('select_categorical', DataFrameSelector(cat_attributes)),
    ('cat_encoder', OneHotEncoder()),
])

full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline),
])

X_train_prepared = full_pipeline.fit_transform(train_after_handling_dicts)
y_train_prepared = train['revenue']

X_train, X_val, y_train, y_val = \
    train_test_split(X_train_prepared, y_train_prepared.values, test_size=0.15, random_state=42)

random_forest = RandomForestRegressor(n_estimators=500, max_depth=10)
random_forest.fit(X_train, y_train)

some_data = train_after_handling_dicts.iloc[:10]
some_labels = train_labels.iloc[:10]
some_data_prepared = full_pipeline.transform(some_data)


some_predictions = random_forest.predict(X_val)[:30]
some_labels = list(y_val)[:30]

differences = np.array(some_predictions) - np.array(some_labels)
relative_errors = differences / np.array(some_labels)

print(relative_errors)


