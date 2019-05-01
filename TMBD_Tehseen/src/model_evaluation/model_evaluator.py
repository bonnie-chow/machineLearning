import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_log_error


class ModelEvaluator:

    def evaluate_model(self, best_forest, input, labels, limit):
        predictions = best_forest.predict(input)
        some_predictions = predictions[:limit]
        some_labels = list(labels)[:limit]

        differences = np.array(some_predictions) - np.array(some_labels)
        relative_errors = differences / np.array(some_labels)
        rmsle = np.sqrt(mean_squared_log_error(list(labels), predictions))

        plt.plot(some_labels, label='true labels', linewidth=2)
        plt.plot(some_predictions, label='predicted labels', linewidth=2)
        plt.show()

        return relative_errors, rmsle

    def find_feature_importance(self, best_forest, cat_attributes, num_attributes):
        return sorted(zip(best_forest.feature_importances_, cat_attributes + num_attributes), reverse=True)
