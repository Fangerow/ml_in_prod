import pandas as pd
from os.path import exists


def create_data():
    """
    :return: data in pd.DataFrame format
    """
    path = 'ml_project/notebooks/heart_cleveland_upload.csv'
    # path = r"C:\Users\User\Desktop\MADE\mlops\shamankov_nikolay\raw_data\heart_cleveland_upload.csv"
    assert exists(path), \
        'required path dataset does not exist'
    data = pd.read_csv(path)
    return data
