import os
import pandas as pd

class ConvenientIO:

    def __init__(self, project_root):
        self.project_root = project_root

    def load_csv_as_DataFrame(self, csv_filename):
        csv_path = os.path.join(self.project_root, 'data', csv_filename)
        return pd.read_csv(csv_path)
