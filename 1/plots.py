"""
Функции постройки диаграмм
"""
import matplotlib.pyplot as plt


def build_cat_plots(dataframe, columns):
    """
    Постройка столбчатых и круговых диаграмм для списка столбцов
    :param dataframe: датафрэйм
    :param columns: список столбцов
    :return:
    """
    for i in columns:
        count = dataframe[i].value_counts()
        figure = plt.figure(figsize=(25, 5))
        ax_1 = figure.add_subplot(1, 2, 1)
        ax_2 = figure.add_subplot(1, 2, 2)

        ax_1.bar(count.index, count.values)
        ax_2.pie(count.values, labels = count.index)
    
def build_num_plots(dataframe, column):
    """
    Постройка гистограммы и графика плотности распределения и диаграммы ящик с усам для столбца
    :param dataframe: датафрэйм
    :param column: столбец
    :return:
    """
    num_data = dataframe[column].dropna()
    figure = plt.figure(figsize=(20, 5))
    ax_1 = figure.add_subplot(1, 3, 1)
    ax_3 = figure.add_subplot(1, 3, 3) 
    ax_2 = figure.add_subplot(1, 3, 2)

    ax_1.hist(num_data)
    num_data.plot.kde()
    ax_3.boxplot(num_data, vert=False)



