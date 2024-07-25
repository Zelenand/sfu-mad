import matplotlib.pyplot as plt
from IPython.display import Image, display
import pydotplus
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.model_selection import GridSearchCV

def draw_param_to_score(ax, train_scores, test_scores, parameter, values):
    """
    Построение графика зависимости счёта от параметра
    :param ax: оси
    :param train_scores: список счётов тренировочной выборки
    :param test_scores: список счётов тестовой выборки
    :param parameter: название параметра
    :param values: список значений параметра
    """
    ax.plot(values, train_scores, label="Train")
    ax.plot(values, test_scores, label="Test")
    ax.set_xlabel(parameter)
    ax.set_ylabel("Score")
    ax.legend(loc='upper left')

def tree_plot(model, columns, file):
    """
    Построение и сохранение в файл диаграммы дерева
    :param model: модель дерева
    :param columns: список столбцов данных
    :param file_name: название файла
    """
    dot_data = export_graphviz(model, feature_names=columns, filled=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png(file)
    display(Image(graph.create_png()))

def build_bar_plot(labels, scores):
    """
    Построение столбчатой диаграммы
    :param labels: подписи столбцов
    :param scores: значения столбцов
    """
    figure = plt.figure(figsize=(20, 10))
    ax = figure.add_subplot(1, 1, 1)
    plt.bar(labels, scores, width=0.8)
    for i, value in enumerate(labels):
        plt.text(value, scores[i], scores[i], ha='center', va='bottom')
    plt.xlabel('Модели')
    plt.ylabel('Оценки')
    plt.show()

def tree_clas_train(parameters, x_train, y_train, x_test, y_test):
    """
    Нахождение лучшей модели DecisionTreeClassifier перебором значений параметров(независимо друг от друга)
    :param parameters: словарь значений параметров
    :param x_train: обучающие данные
    :param y_train: целевые данные обучающей выборки
    :param x_test: тестовые данные
    :param y_test:целевые данные тестовой выборки
    :return: лучшая модель DecisionTreeClassifier
    """
    best_parameters = {
        'max_depth': None,
        'min_samples_split': None,
        'min_samples_leaf': None,
        'max_leaf_nodes': None
    }
    fig, ax = plt.subplots(figsize=(20, 40), nrows=4, ncols=1)
    fig.suptitle(f"Зависимость счёта дерева решений от параметров")
    i = 0
    for parameter in parameters.keys():
        models = []
        train_scores = []
        test_scores = []
        for value in parameters[parameter]:
            if parameter == "max_depth":
                model = DecisionTreeClassifier(random_state=42, max_depth=value)
            elif parameter == 'min_samples_split':
                model = DecisionTreeClassifier(random_state=42, min_samples_split=value)
            elif parameter == 'min_samples_leaf':
                model = DecisionTreeClassifier(random_state=42, min_samples_leaf=value)
            elif parameter == 'max_leaf_nodes':
                model = DecisionTreeClassifier(random_state=42, max_leaf_nodes=value)
            model.fit(x_train, y_train)
            models.append(model)
            y_train_predict = model.predict_proba(x_train)[:, 1]
            y_test_predict = model.predict_proba(x_test)[:, 1]
            train_scores.append(roc_auc_score(y_train, y_train_predict))
            test_scores.append(roc_auc_score(y_test, y_test_predict))
        index_of_best = test_scores.index(max(test_scores))
        best_model = models[index_of_best]
        best_parameters[parameter] = best_model.get_params()[parameter]
        draw_param_to_score(ax[i], train_scores, test_scores, parameter, parameters[parameter])
        i += 1
        print("Лучшая модель с параметром", parameter, best_model)
        print("Результат: {:.4f}".format(test_scores[index_of_best]))
    best_model = DecisionTreeClassifier(
        random_state=42,
        max_depth=best_parameters['max_depth'],
        min_samples_split=best_parameters['min_samples_split'],
        min_samples_leaf=best_parameters['min_samples_leaf'],
        max_leaf_nodes=best_parameters['max_leaf_nodes'])
    best_model.fit(x_train, y_train)
    y_test_predict = model.predict_proba(x_test)[:, 1]
    best_model_score = roc_auc_score(y_test, y_test_predict)
    print("Лучшая модель с 4 параметрами", best_model)
    print("Результат: {:.4f}".format(best_model_score))
    return best_model


def tree_regr_train(parameters, x_train, y_train, x_test, y_test):
    """
    Нахождение лучшей модели DecisionTreeRegressor перебором значений параметров(независимо друг от друга)
    :param parameters: словарь значений параметров
    :param x_train: обучающие данные
    :param y_train: целевые данные обучающей выборки
    :param x_test: тестовые данные
    :param y_test:целевые данные тестовой выборки
    :return: лучшая модель DecisionTreeRegressor
    """
    best_parameters = {
        'max_depth': None,
        'min_samples_split': None,
        'min_samples_leaf': None,
        'max_leaf_nodes': None
    }
    fig, ax = plt.subplots(figsize=(20, 40), nrows=4, ncols=1)
    fig.suptitle(f"Зависимость счёта дерева решений от параметров")
    i = 0
    for parameter in parameters.keys():
        models = []
        train_scores = []
        test_scores = []
        for value in parameters[parameter]:
            if parameter == "max_depth":
                model = DecisionTreeRegressor(random_state=42, max_depth=value)
            elif parameter == 'min_samples_split':
                model = DecisionTreeRegressor(random_state=42, min_samples_split=value)
            elif parameter == 'min_samples_leaf':
                model = DecisionTreeRegressor(random_state=42, min_samples_leaf=value)
            elif parameter == 'max_leaf_nodes':
                model = DecisionTreeRegressor(random_state=42, max_leaf_nodes=value)
            model.fit(x_train, y_train)
            models.append(model)
            y_train_predict = model.predict(x_train)
            y_test_predict = model.predict(x_test)
            train_scores.append(mean_squared_error(y_train, y_train_predict, squared=False))
            test_scores.append(mean_squared_error(y_test, y_test_predict, squared=False))
        index_of_best = test_scores.index(min(test_scores))
        best_model = models[index_of_best]
        best_parameters[parameter] = best_model.get_params()[parameter]
        draw_param_to_score(ax[i], train_scores, test_scores, parameter, parameters[parameter])
        i += 1
        print("Лучшая модель с параметром", parameter, best_model)
        print("Результат: {:.4f}".format(test_scores[index_of_best]))
    best_model = DecisionTreeRegressor(
        random_state=42,
        max_depth=best_parameters['max_depth'],
        min_samples_split=best_parameters['min_samples_split'],
        min_samples_leaf=best_parameters['min_samples_leaf'],
        max_leaf_nodes=best_parameters['max_leaf_nodes'])
    best_model.fit(x_train, y_train)
    y_test_predict = model.predict(x_test)
    best_model_score = mean_squared_error(y_test, y_test_predict, squared=False)
    print("Лучшая модель с 4 параметрами", best_model)
    print("Результат: {:.4f}".format(best_model_score))
    return best_model

def tree_clas_train_2(parameters, x_train, y_train, x_test, y_test):
    """
    Нахождение лучшей модели DecisionTreeClassifier перебором значений параметров(последовательно,
    учитывая лучшее значение предыдущих параметров)
    :param parameters: словарь значений параметров
    :param x_train: обучающие данные
    :param y_train: целевые данные обучающей выборки
    :param x_test: тестовые данные
    :param y_test:целевые данные тестовой выборки
    :return: лучшая модель DecisionTreeClassifier
    """
    best_parameters = {
        'max_depth': None,
        'min_samples_split': None,
        'min_samples_leaf': None,
        'max_leaf_nodes': None
    }
    i = 0
    for parameter in parameters.keys():
        models = []
        train_scores = []
        test_scores = []
        for value in parameters[parameter]:
            if parameter == "max_depth":
                model = DecisionTreeClassifier(random_state=42, max_depth=value)
            elif parameter == 'min_samples_split':
                model = DecisionTreeClassifier(random_state=42, max_depth=best_parameters['max_depth'],
                                               min_samples_split=value)
            elif parameter == 'min_samples_leaf':
                model = DecisionTreeClassifier(random_state=42, max_depth=best_parameters['max_depth'],
        min_samples_split=best_parameters['min_samples_split'], min_samples_leaf=value)
            elif parameter == 'max_leaf_nodes':
                model = DecisionTreeClassifier(random_state=42, max_depth=best_parameters['max_depth'],
        min_samples_split=best_parameters['min_samples_split'],
        min_samples_leaf=best_parameters['min_samples_leaf'], max_leaf_nodes=value)
            model.fit(x_train, y_train)
            models.append(model)
            y_train_predict = model.predict_proba(x_train)[:, 1]
            y_test_predict = model.predict_proba(x_test)[:, 1]
            train_scores.append(roc_auc_score(y_train, y_train_predict))
            test_scores.append(roc_auc_score(y_test, y_test_predict))
        index_of_best = test_scores.index(max(test_scores))
        best_model = models[index_of_best]
        best_parameters[parameter] = best_model.get_params()[parameter]
        i += 1
    best_model = DecisionTreeClassifier(
        random_state=42,
        max_depth=best_parameters['max_depth'],
        min_samples_split=best_parameters['min_samples_split'],
        min_samples_leaf=best_parameters['min_samples_leaf'],
        max_leaf_nodes=best_parameters['max_leaf_nodes'])
    best_model.fit(x_train, y_train)
    y_test_predict = model.predict_proba(x_test)[:, 1]
    best_model_score = roc_auc_score(y_test, y_test_predict)
    return best_model


def tree_regr_train_2(parameters, x_train, y_train, x_test, y_test):
    """
    Нахождение лучшей модели DecisionTreeRegressor перебором значений параметров(последовательно,
    учитывая лучшее значение предыдущих параметров)
    :param parameters: словарь значений параметров
    :param x_train: обучающие данные
    :param y_train: целевые данные обучающей выборки
    :param x_test: тестовые данные
    :param y_test: целевые данные тестовой выборки
    :return: лучшая модель DecisionTreeRegressor
    """
    best_parameters = {
        'max_depth': None,
        'min_samples_split': None,
        'min_samples_leaf': None,
        'max_leaf_nodes': None
    }
    i = 0
    for parameter in parameters.keys():
        models = []
        train_scores = []
        test_scores = []
        for value in parameters[parameter]:
            if parameter == "max_depth":
                model = DecisionTreeRegressor(random_state=42, max_depth=value)
            elif parameter == 'min_samples_split':
                model = DecisionTreeRegressor(random_state=42, max_depth=best_parameters['max_depth'], min_samples_split=value)
            elif parameter == 'min_samples_leaf':
                model = DecisionTreeRegressor(random_state=42, max_depth=best_parameters['max_depth'],
        min_samples_split=best_parameters['min_samples_split'], min_samples_leaf=value)
            elif parameter == 'max_leaf_nodes':
                model = DecisionTreeRegressor(random_state=42, max_depth=best_parameters['max_depth'],
        min_samples_split=best_parameters['min_samples_split'],
        min_samples_leaf=best_parameters['min_samples_leaf'], max_leaf_nodes=value)
            model.fit(x_train, y_train)
            models.append(model)
            y_train_predict = model.predict(x_train)
            y_test_predict = model.predict(x_test)
            train_scores.append(mean_squared_error(y_train, y_train_predict, squared=False))
            test_scores.append(mean_squared_error(y_test, y_test_predict, squared=False))
        index_of_best = test_scores.index(min(test_scores))
        best_model = models[index_of_best]
        best_parameters[parameter] = best_model.get_params()[parameter]
        i += 1
    best_model = DecisionTreeRegressor(
        random_state=42,
        max_depth=best_parameters['max_depth'],
        min_samples_split=best_parameters['min_samples_split'],
        min_samples_leaf=best_parameters['min_samples_leaf'],
        max_leaf_nodes=best_parameters['max_leaf_nodes'])
    best_model.fit(x_train, y_train)
    y_test_predict = model.predict(x_test)
    best_model_score = mean_squared_error(y_test, y_test_predict, squared=False)
    return best_model

def grid_search_clas(params, X_train, y_train):
    model = DecisionTreeClassifier(random_state=42)
    g_s = GridSearchCV(model, params, cv=10, n_jobs=4, scoring='roc_auc')

    cv = g_s.fit(X_train, y_train)

    print("Tuned hpyerparameters (best parameters):", cv.best_params_)

    return cv.best_params_

def grid_search_regr(params, X_train, y_train):
    model = DecisionTreeRegressor(random_state=42)
    g_s = GridSearchCV(model, params, cv=10, n_jobs=4, scoring='neg_root_mean_squared_error')

    cv = g_s.fit(X_train, y_train)

    print("Tuned hpyerparameters (best parameters):", cv.best_params_)

    return cv.best_params_

def regr_tree_pruning(model, ccp_alphas, x_train, y_train, x_test, y_test):
    """
    Нахождение лучшей обрезки модели DecisionTreeRegressor перебором значений ccp_alpha
    :param parameters: словарь значений параметров
    :param x_train: обучающие данные
    :param y_train: целевые данные обучающей выборки
    :param x_test: тестовые данные
    :param y_test: целевые данные тестовой выборки
    :return: лучшая модель DecisionTreeRegressor, счёт лучшей модели
    """
    models = []
    test_scores = []
    r2s = []
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1)
    fig.suptitle("Изменение счёта при обрезке дерева")
    for value in ccp_alphas:
        param = {'ccp_alpha': value}
        model = model.set_params(**param)
        models.append(model)
        model.fit(x_train, y_train)
        y_pred_test = model.predict(x_test)
        test_scores.append(mean_squared_error(
            y_test, y_pred_test, squared=False))
        r2s.append(r2_score(y_test, y_pred_test))
    index_of_best = test_scores.index(min(test_scores))
    best_model = models[index_of_best]
    ax.plot(ccp_alphas, test_scores)
    ax.set_xlabel("ccp_alpha")
    ax.set_ylabel("Score")
    return best_model, test_scores[index_of_best], r2s[index_of_best]


def clas_tree_pruning(model, ccp_alphas, x_train, y_train, x_test, y_test):
    """
    Нахождение лучшей обрезки модели DecisionTreeClassifier перебором значений ccp_alpha
    :param parameters: словарь значений параметров
    :param x_train: обучающие данные
    :param y_train: целевые данные обучающей выборки
    :param x_test: тестовые данные
    :param y_test: целевые данные тестовой выборки
    :return: лучшая модель DecisionTreeClassifier, счёт лучшей модели
    """
    models = []
    test_scores = []
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1)
    fig.suptitle("Изменение счёта при обрезке дерева")
    for value in ccp_alphas:
        param = {'ccp_alpha': value}
        model = model.set_params(**param)
        models.append(model)
        model.fit(x_train, y_train)
        y_pred_test = model.predict_proba(x_test)[:, 1]
        test_scores.append(roc_auc_score(y_test, y_pred_test))
    index_of_best = test_scores.index(max(test_scores))
    best_model = models[index_of_best]
    ax.plot(ccp_alphas, test_scores)
    ax.set_xlabel("ccp_alpha")
    ax.set_ylabel("Score")
    return best_model, test_scores[index_of_best]