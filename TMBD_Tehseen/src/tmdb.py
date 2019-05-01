import warnings

import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import SGD

from src.data_preparation.column_data_preprocessor import ColumnDataPreprocessor
from src.data_preparation.column_selector import ColumnSelector
from src.data_preparation.data_frame_selector import DataFrameSelector
from src.data_preparation.dictionary_data_extractor import DictionaryDataExtractor
from src.ml_models.dense_layer_model import DenseLayerModel
from src.model_evaluation.model_evaluator import ModelEvaluator

warnings.filterwarnings('ignore')

dictionary_data_extractor = DictionaryDataExtractor()
column_selector = ColumnSelector()
column_data_preprocessor = ColumnDataPreprocessor()
model_evaluator = ModelEvaluator()

train = pd.read_csv("../resources/train.csv")
test = pd.read_csv("../resources/test.csv")

train = dictionary_data_extractor.text_to_dict(train)
test = dictionary_data_extractor.text_to_dict(test)

train_without_id = column_selector.remove_unwanted_columns(train, ['id'])

reduced_train_set = column_selector.consider_subset_of_columns(train_without_id, ['budget', 'genres', 'original_language', 'popularity', 'release_date', 'runtime', 'Keywords', 'cast', 'crew','revenue'])

reduced_train_set['cleansed_languages'] = str(column_data_preprocessor.preprocess_languages(reduced_train_set))
reduced_train_set['release_year'] = dictionary_data_extractor.extract_release_year(reduced_train_set)
reduced_train_set['main_genre'] = str(dictionary_data_extractor.extract_data_from_dict(reduced_train_set, 'genres', 'name'))
reduced_train_set['main_actor'] = str(dictionary_data_extractor.extract_data_from_dict(reduced_train_set, 'cast', 'name'))
reduced_train_set['main_actor'] = str(dictionary_data_extractor.convert_if_condition_is_met(reduced_train_set, 'main_actor', 3))
reduced_train_set['main_keyword'] = str(dictionary_data_extractor.extract_data_from_dict(reduced_train_set, 'Keywords', 'name'))
reduced_train_set['main_keyword'] = str(dictionary_data_extractor.convert_if_condition_is_met(reduced_train_set, 'main_keyword', 6))
reduced_train_set['main_crew_member'] = str(dictionary_data_extractor.extract_data_from_dict(reduced_train_set, 'crew', 'name'))
reduced_train_set['main_crew_member'] = str(dictionary_data_extractor.convert_if_condition_is_met(reduced_train_set, 'main_crew_member', 6))

train_after_handling_dicts = column_selector.remove_unwanted_columns(reduced_train_set, ['original_language', 'genres', 'Keywords', 'cast', 'crew', 'release_date'])

train_labels = train_after_handling_dicts['revenue']

cat_attributes = ['cleansed_languages', 'main_genre', 'main_keyword', 'main_actor', 'main_crew_member']
num_attributes = ['budget', 'popularity', 'runtime', 'release_year']

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
    train_test_split(X_train_prepared, y_train_prepared.values, test_size=0.1, random_state=42)

print('shape of training set:')
print(type(X_train))
print(X_train.shape)

param_grid = [
    {'n_estimators': [3, 5, 10], 'max_depth': [5, 10, 15], 'max_features': [3, 6, 9]}
]

random_forest = RandomForestRegressor()
grid_search = GridSearchCV(random_forest, param_grid, cv=5, scoring='neg_mean_squared_log_error')
grid_search.fit(X_train, y_train)

print('The best parameters are:')
print(grid_search.best_params_)

best_forest = grid_search.best_estimator_
print(model_evaluator.find_feature_importance(best_forest, cat_attributes, num_attributes))

best_forest.fit(X_train, y_train)

relative_validation_errors, rmsle_validation = model_evaluator.evaluate_model(best_forest, X_val, y_val, 40)
relative_train_errors, rmsle_train = model_evaluator.evaluate_model(best_forest, X_train, y_train, 40)
print('relative validation errors:')
print('validation:', relative_validation_errors)
print('train', relative_train_errors)
print('rmsle error of the model:')
print('validation:', rmsle_validation)
print('train:', rmsle_train)

#random_forest

# dense_layer_model = DenseLayerModel()
#
# model = dense_layer_model.build_model()
# print('model succesfully created')
# optimizer = SGD(lr=0.001)
# learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
#
# model.compile(optimizer=optimizer, loss="mean_squared_logarithmic_error", metrics=["accuracy"])
# print('model succesfully compiled')
# model.fit(X_train, y_train,
#           batch_size=20, epochs=50, validation_data=(X_val, y_val), callbacks=[learning_rate_reduction])
# print('model succesfully trained')
# predictions = model.predict(X_val)
