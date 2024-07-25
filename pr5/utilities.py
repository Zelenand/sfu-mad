import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import StackingRegressor, StackingClassifier, BaggingClassifier, BaggingRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

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

def train_begging_clas(parameters, g_s_parameters, x_train, y_train, x_test, y_test):
    fig, ax = plt.subplots(figsize=(20, 40), nrows=len(parameters), ncols=1)
    fig.suptitle(f"Зависимость счёта ансамбля BaggingClassifier от параметров")
    i = 0
    for parameter in parameters.keys():
        models = []
        train_scores = []
        test_scores = []
        for value in parameters[parameter]:
            temp_params = {str(parameter): value}
            model = BaggingClassifier(random_state=0).set_params(**temp_params)
            model.fit(x_train, y_train)
            models.append(model)
            y_train_predict = model.predict_proba(x_train)[:, 1]
            y_test_predict = model.predict_proba(x_test)[:, 1]
            train_scores.append(roc_auc_score(y_train, y_train_predict))
            test_scores.append(roc_auc_score(y_test, y_test_predict))
        index_of_best = test_scores.index(max(test_scores))
        best_model = models[index_of_best]
        draw_param_to_score(ax[i], train_scores, test_scores, parameter, parameters[parameter])
        i += 1
        print("Лучшая модель с параметром", parameter, best_model)
        print("Результат: {:.4f}".format(test_scores[index_of_best]))
    model = BaggingClassifier(random_state=0)
    best_model = grid_search(g_s_parameters, model, x_train, y_train, scoring='roc_auc')
    y_test_predict = best_model.predict_proba(x_test)[:, 1]
    best_model_score = roc_auc_score(y_test, y_test_predict)
    print("Лучшая модель с параметрами", best_model.best_params_)
    print("Результат: {:.4f}".format(best_model_score))

    return best_model

def train_begging_regr(parameters, g_s_parameters, x_train, y_train, x_test, y_test):
    fig, ax = plt.subplots(figsize=(20, 40), nrows=len(parameters), ncols=1)
    fig.suptitle(f"Зависимость счёта ансамбля BaggingRegressor от параметров")
    i = 0
    for parameter in parameters.keys():
        models = []
        train_scores = []
        test_scores = []
        for value in parameters[parameter]:
            temp_params = {str(parameter): value}
            model = BaggingRegressor(random_state=0).set_params(**temp_params)
            model.fit(x_train, y_train)
            models.append(model)
            y_train_predict = model.predict(x_train)
            y_test_predict = model.predict(x_test)
            train_scores.append(mean_squared_error(y_train, y_train_predict, squared=False))
            test_scores.append(mean_squared_error(y_test, y_test_predict, squared=False))
        index_of_best = test_scores.index(min(test_scores))
        best_model = models[index_of_best]
        draw_param_to_score(ax[i], train_scores, test_scores, parameter, parameters[parameter])
        i += 1
        print("Лучшая модель с параметром", parameter, best_model)
        print("Результат: {:.4f}".format(test_scores[index_of_best]))
    model = BaggingRegressor(random_state=0)
    best_model = grid_search(g_s_parameters, model, x_train, y_train, scoring='neg_root_mean_squared_error')
    y_test_predict = best_model.predict(x_test)
    best_model_score = mean_squared_error(y_test, y_test_predict, squared=False)
    print("Лучшая модель с параметрами", best_model.best_params_)
    print("Результат: {:.4f}".format(best_model_score))

    return best_model

def train_boosting_clas(parameters, g_s_parameters, x_train, y_train, x_test, y_test):
    fig, ax = plt.subplots(figsize=(20, 40), nrows=len(parameters), ncols=1)
    fig.suptitle(f"Зависимость счёта ансамбля GradientBoostingClassifier от параметров")
    i = 0
    for parameter in parameters.keys():
        models = []
        train_scores = []
        test_scores = []
        for value in parameters[parameter]:
            temp_params = {str(parameter): value}
            model = GradientBoostingClassifier(random_state=0).set_params(**temp_params)
            model.fit(x_train, y_train)
            models.append(model)
            y_train_predict = model.predict_proba(x_train)[:, 1]
            y_test_predict = model.predict_proba(x_test)[:, 1]
            train_scores.append(roc_auc_score(y_train, y_train_predict))
            test_scores.append(roc_auc_score(y_test, y_test_predict))
        index_of_best = test_scores.index(max(test_scores))
        best_model = models[index_of_best]
        draw_param_to_score(ax[i], train_scores, test_scores, parameter, parameters[parameter])
        i += 1
        print("Лучшая модель с параметром", parameter, best_model)
        print("Результат: {:.4f}".format(test_scores[index_of_best]))
    model = GradientBoostingClassifier(random_state=0)
    best_model = grid_search(g_s_parameters, model, x_train, y_train, scoring='roc_auc')
    y_test_predict = best_model.predict_proba(x_test)[:, 1]
    best_model_score = roc_auc_score(y_test, y_test_predict)
    print("Лучшая модель с параметрами", best_model.best_params_)
    print("Результат: {:.4f}".format(best_model_score))

    return best_model

def train_boosting_regr(parameters, g_s_parameters, x_train, y_train, x_test, y_test):
    fig, ax = plt.subplots(figsize=(20, 40), nrows=len(parameters), ncols=1)
    fig.suptitle(f"Зависимость счёта ансамбля GradientBoostingRegressor от параметров")
    i = 0
    for parameter in parameters.keys():
        models = []
        train_scores = []
        test_scores = []
        for value in parameters[parameter]:
            temp_params = {str(parameter): value}
            model = GradientBoostingRegressor(random_state=0).set_params(**temp_params)
            model.fit(x_train, y_train)
            models.append(model)
            y_train_predict = model.predict(x_train)
            y_test_predict = model.predict(x_test)
            train_scores.append(mean_squared_error(y_train, y_train_predict, squared=False))
            test_scores.append(mean_squared_error(y_test, y_test_predict, squared=False))
        index_of_best = test_scores.index(min(test_scores))
        best_model = models[index_of_best]
        draw_param_to_score(ax[i], train_scores, test_scores, parameter, parameters[parameter])
        i += 1
        print("Лучшая модель с параметром", parameter, best_model)
        print("Результат: {:.4f}".format(test_scores[index_of_best]))
    model = GradientBoostingRegressor(random_state=0)
    best_model = grid_search(g_s_parameters, model, x_train, y_train, scoring='neg_root_mean_squared_error')
    y_test_predict = best_model.predict(x_test)
    best_model_score = mean_squared_error(y_test, y_test_predict, squared=False)
    print("Лучшая модель с параметрами", best_model.best_params_)
    print("Результат: {:.4f}".format(best_model_score))

    return best_model

def train_stacking_clas(x_train, y_train, x_test, y_test):
    models = ['KNeighborsClassifier', 'SVC', 'GaussianNB', 'LogisticRegression']
    trained_models = []
    train_scores = []
    test_scores = []
    models_models = []
    for i in range(2, len(models) + 1):
        models_combinations = combinations(models, i)
        for cur_models_names in models_combinations:
            cur_models = []
            for name in cur_models_names:
                if name == 'KNeighborsClassifier': cur_models.append(('KNeighborsClassifier', KNeighborsClassifier()))
                elif name == 'SVC': cur_models.append(('SVC', SVC()))
                elif name == 'GaussianNB':cur_models.append(('GaussianNB', GaussianNB()))
                elif name == 'LogisticRegression': cur_models.append(('LogisticRegression', LogisticRegression()))
            model = StackingClassifier(estimators=cur_models, final_estimator=LogisticRegression())
            model.fit(x_train, y_train)
            trained_models.append(model)
            y_train_predict = model.predict_proba(x_train)[:, 1]
            y_test_predict = model.predict_proba(x_test)[:, 1]
            train_score = roc_auc_score(y_train, y_train_predict)
            test_score = roc_auc_score(y_test, y_test_predict)
            train_scores.append(train_score)
            test_scores.append(test_score)
            models_models.append(cur_models)
            print("Результат для", cur_models_names, test_score)
    index_of_best = test_scores.index(max(test_scores))
    best_model = trained_models[index_of_best]
    print("Лучшая модель: ", models_models[index_of_best], test_scores[index_of_best])
    return best_model

def train_stacking_regr(x_train, y_train, x_test, y_test):
    models = ['KNeighborsRegressor', 'SVR', 'DecisionTreeRegressor']
    trained_models = []
    train_scores = []
    test_scores = []
    models_models = []
    for i in range(2, len(models) + 1):
        models_combinations = combinations(models, i)
        for cur_models_names in models_combinations:
            cur_models = []
            for name in cur_models_names:
                if name == 'KNeighborsRegressor':
                    cur_models.append(('KNeighborsRegressor', KNeighborsRegressor()))
                elif name == 'SVR':
                    cur_models.append(('SVR', SVR()))
                elif name == 'DecisionTreeRegressor':
                    cur_models.append(('DecisionTreeRegressor', DecisionTreeRegressor()))
            model = StackingRegressor(estimators=cur_models, final_estimator=LinearRegression())
            model.fit(x_train, y_train)
            trained_models.append(model)
            y_train_predict = model.predict(x_train)
            y_test_predict = model.predict(x_test)
            train_score = mean_squared_error(y_train, y_train_predict, squared=False)
            test_score = mean_squared_error(y_test, y_test_predict, squared=False)
            train_scores.append(train_score)
            test_scores.append(test_score)
            models_models.append(cur_models)
            print("Результат для", cur_models_names, test_score)
    index_of_best = test_scores.index(min(test_scores))
    best_model = trained_models[index_of_best]
    print("Лучшая модель: ", models_models[index_of_best], test_scores[index_of_best])
    return best_model

def grid_search(params, model, X_train, y_train, scoring):
    g_s = GridSearchCV(model, params, cv=10, n_jobs=-1, scoring=scoring)

    cv = g_s.fit(X_train, y_train)

    print("Tuned hpyerparameters (best parameters):", cv.best_params_)

    return cv
