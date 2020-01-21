import pandas as pd

# text data in the <text> column
def getData(path, columnName):
    data = pd.read_csv(path, error_bad_lines=False)
    data_text = data[[columnName]]
    data_text['index'] = data_text.index
    return data_text