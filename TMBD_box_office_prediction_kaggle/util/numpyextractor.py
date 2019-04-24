import pandas
import numpy
from functools import reduce


class NumpyExtractor:
    _acceptedtypes = {
        pandas.core.frame.DataFrame: 'PANDAS_DF',
        pandas.core.series.Series: 'PANDAS_SR',
        numpy.ndarray: 'NUMPY'}

    def __init__(self, data):
        self._raw_data = data
        self.data_type = self._check_type(data)

    def _check_type(self, data):
        if self._validate_type(data):
            return NumpyExtractor._acceptedtypes[type(data)]
        else:
            raise AttributeError('Data type not accepted')

    def _validate_type(self, data):
        return reduce(lambda bool1, bool2: bool1 or bool2,
                      [isinstance(data, allowed_typ) for allowed_typ in NumpyExtractor._acceptedtypes.keys()])

    def extract(self):
        if self.data_type == "NUMPY":
            return self._raw_data
        if self.data_type == "PANDAS_DF":
            return self._raw_data.values
        if self.data_type == "PANDAS_SR":
            print("Series case ->", self._raw_data.values.reshape(-1, 1) )
            return self._raw_data.values.reshape(-1, 1)
