class ColumnSelector():

    def remove_unwanted_columns(self, df, column_to_remove):
        return df.drop(columns=column_to_remove)

    def consider_subset_of_columns(self, df, columns):
        return df[columns]

