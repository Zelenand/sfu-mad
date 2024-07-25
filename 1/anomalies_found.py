"""
Функции реализующие методы поиска выбросов в данных
"""
import numpy as np
import scipy

def quartile_method(dataframe, collumn):
    """
    Метод квартилей для нахождения выбросов
    :param dataframe: датафрэйм
    :param collumn: название столбца
    :return: датафрэйм после использования метода квартилей
    """
    Q1 = dataframe[collumn].quantile(0.25)
    Q3 = dataframe[collumn].quantile(0.75)
    lower = Q1 - 1.5 * (Q3 - Q1)
    upper = Q3 + 1.5 * (Q3 - Q1)
    print("Границы для", collumn, lower, upper)
    dataframe = dataframe[
        (np.isnan(dataframe[collumn])) | (dataframe[collumn] < upper) & (
                    dataframe[collumn] > lower)]
    return dataframe

def sigma_method(dataframe, collumn):
    """
    Метод сигм для нахождения выбросов
    :param dataframe: датафрэйм
    :param collumn: название столбца
    :return: датафрэйм после использования метода сигм
    """
    df_without_nan = dataframe[np.isnan(dataframe[collumn]) == False]
    clipped, lower, upper = scipy.stats.sigmaclip(df_without_nan[collumn], 2, 2)
    print("Границы для", collumn, lower, upper)
    dataframe = dataframe[(np.isnan(dataframe[collumn])) | (dataframe[collumn] < upper) & (dataframe[collumn] > lower)]
    return dataframe

