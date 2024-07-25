from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np


def inf_atr(method, x, y, cols = None):
    selected_features = cols
    if method is not None:
        method.fit(x, y)
        cols_idxs = method.get_support(indices=True)
        x = x.iloc[:, cols_idxs]
        selected_features = x.columns
        x = x[selected_features]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)
    gnb = LogisticRegression()
    y_pred = gnb.fit(X_train, y_train).predict_proba(X_test)
    score = roc_auc_score(y_test, y_pred[:, 1])

    return selected_features, score


def grid_search(model, params, X_train, X_test, y_train, y_test):
    g_s = GridSearchCV(model, params, cv=10, n_jobs=4, scoring='roc_auc')

    cv = g_s.fit(X_train, y_train)

    print("Tuned hpyerparameters (best parameters):", cv.best_params_)
    print("Train accuracy:", cv.best_score_ // 0.0001 / 10000)
    print("Test accuracy:", cv.best_estimator_.score(X_test, y_test) // 0.0001 / 10000)

    return cv


def build_cat_plots(dataframe, columns):
    for i in columns:
        count = dataframe[i].value_counts()
        figure = plt.figure(figsize=(25, 5))
        ax_1 = figure.add_subplot(1, 2, 1)
        ax_2 = figure.add_subplot(1, 2, 2)

        ax_1.bar(count.index, count.values)
        ax_2.pie(count.values, labels=count.index)


def sampling(method, x, y, title):
    import pandas as pd
    x_new, y_new = method.fit_resample(x, y)
    count = pd.DataFrame(y_new)["Churn"].value_counts()
    figure = plt.figure(figsize=(25, 5))
    ax_1 = figure.add_subplot(1, 2, 1)
    ax_1.set_title(title)
    ax_1.pie(count.values, labels=count.index)

    X_train, X_test, y_train, y_test = train_test_split(x_new, y_new, test_size=0.5, random_state=0)
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict_proba(X_test)
    score = roc_auc_score(y_test, y_pred[:, 1])
    print(score // 0.0001 / 10000)
    return x_new, y_new, score


def build_bar_plot(labels, scores):
    plt.bar(labels, scores, width=0.8)
    for i, value in enumerate(labels):
        plt.text(value, scores[i], scores[i], ha='center', va='bottom')
    plt.xlabel('Модели')
    plt.ylabel('Оценки')
    plt.show()


def roc_auc_plot(models, X_test, Y_test):
    plt.figure(figsize=(20, 7))
    test_score = []
    for model in models:
        Y_test_pred = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(Y_test, Y_test_pred)
        plt.plot(*roc_curve(Y_test, Y_test_pred)
                 [:2], label=f'{model.best_estimator_.__class__.__name__}=' +
                 '{: .4f}'.format(test_auc))
        test_score.append(test_auc)
    legend_box = plt.legend(fontsize='large', framealpha=1).get_frame()
    legend_box.set_facecolor("white")
    legend_box.set_edgecolor("black")
    plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    plt.title("ROC кривые")
    plt.show()

def build_param_score_plot(x, y):
    plt.plot(x, y)
    plt.show()
