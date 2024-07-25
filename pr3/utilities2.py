from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import ttest_ind
from scipy.stats import t
from scipy.stats import sem

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
    g_s = GridSearchCV(model, params, cv=10, n_jobs=4, scoring='neg_root_mean_squared_error')

    cv = g_s.fit(X_train, y_train)

    print("Tuned hpyerparameters (best parameters):", cv.best_params_)

    return cv


def build_cat_plots(dataframe, columns):
    for i in columns:
        count = dataframe[i].value_counts()
        figure = plt.figure(figsize=(30, 5))
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


def build_bar_plot(labels, scores, title):
    figure = plt.figure(figsize=(25, 5))
    plt.bar(labels, scores, width=1)
    for i, value in enumerate(labels):
        plt.text(value, scores[i], scores[i], ha='center', va='bottom')
    plt.xlabel('Модели')
    plt.ylabel('Оценки')
    plt.title(title)
    plt.show()

def build_param_score_plot(x, y):
    plt.plot(x, y)
    plt.show()

def get_aic(n, rmse, num_params):
    return n * np.log(rmse**2) + 2 * num_params

def get_bic(n, rmse, num_params):
    return n * np.log(rmse**2) + np.log(n) * num_params

def get_stats(model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_pred = pd.DataFrame(y_pred, index=y_test.index)
    y_test = pd.DataFrame(y_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False) // 0.1 / 10
    r2 = r2_score(y_test, y_pred)  // 0.0001 / 10000
    adj_r2 = 1 - (1 - r2_score(y_test, y_pred)) * ((len(y_test) - 1) / (len(y_test) - x_test.shape[1] - 1))
    aic = get_aic(y_test.shape[0], rmse, x_test.shape[1])
    bic = get_bic(y_test.shape[0], rmse, x_test.shape[1])
    tval, pval = ttest_ind(y_test, y_pred)
    interval_95 = t.interval(alpha=0.95, df=y_pred.shape[0]-1,
              loc=y_pred.mean(),
              scale=sem(y_pred))
    print('RMSE of model: {:.4f}'.format(rmse))
    print('R2 of model: {:.4f}'.format(r2))
    print('Adj R2 of model: {:.4f}'.format(adj_r2))
    print('AIC of model: {:.4f}'.format(aic))
    print('BIC of model: {:.4f}'.format(bic))

    return rmse, r2

def param_stats(model, x_test, y_test):
    model.fit(x_test, y_test)
    if hasattr(model, 'intercept_') and hasattr(model, 'coef_'):
        params = np.append(model.intercept_, model.coef_)
    else:
        return
    predictions = model.predict(x_test)
    newX = pd.DataFrame({"Constant": np.ones(len(x_test))}, index=x_test.index).join(pd.DataFrame(x_test))
    MSE = (sum((y_test - predictions) ** 2)) / (len(newX) - len(newX.columns))

    var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params / sd_b

    p_values = [2 * (1 - t.cdf(np.abs(i), (len(newX) - x_test.shape[1]))) for i in ts_b]
    intervals = [t.interval(alpha=0.95, df=x.shape[0]-1, loc=x.mean(), scale=sem(x)) for _, x in  newX.iteritems()]
    intervals_a = [intervals[x][0] for x in range(0, len(intervals))]
    intervals_b = [intervals[x][1] for x in range(0, len(intervals))]
    sd_b = np.round(sd_b, 3)
    ts_b = np.round(ts_b, 3)
    p_values = np.round(p_values, 3)
    params = np.round(params, 4)

    myDF3 = pd.DataFrame()
    myDF3['param'], myDF3["Coefficients"], myDF3["Standard Errors"], myDF3["t values"], myDF3["P> |t|"], myDF3["interval_a"], myDF3["interval_b"] = \
        [newX.columns, params, sd_b, ts_b, p_values, intervals_a, intervals_b]
    print(myDF3[1:10])