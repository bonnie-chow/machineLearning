from unittest import TestCase
import numpy as np
import pandas as pd

from numpyextractor import NumpyExtractor


class TestNumpyExtractor(TestCase):

    def test_works_with_numpy_aray(self):
        array = np.array([[1, 2, 3], [4, 5, 6]])
        extractor = NumpyExtractor(array)
        self.assertEqual(extractor.data_type, "NUMPY")
        expected = array
        np.testing.assert_array_equal(expected, extractor.extract())

    def test_works_with_data_frame(self):
        data_frame = pd.DataFrame({"values": [1, 2, 3, 4, 5], "another_values": [6, 7, 8, 9, 0]})
        extractor = NumpyExtractor(data_frame)
        self.assertEqual(extractor.data_type, "PANDAS_DF")
        expected = np.array([[1, 6], [2, 7], [3, 8], [4, 9], [5, 0]])
        np.testing.assert_array_equal(expected, extractor.extract())

    def test_works_with_series(self):
        data_frame = pd.DataFrame({"values": [1, 2, 3, 4, 5]})
        series = data_frame["values"]
        extractor = NumpyExtractor(series)
        self.assertEqual(extractor.data_type, "PANDAS_SR")
        expected = np.array([[1], [2], [3], [4], [5]])
        np.testing.assert_array_equal(expected, extractor.extract())


1
