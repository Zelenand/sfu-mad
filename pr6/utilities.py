import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score, completeness_score,v_measure_score, silhouette_score
from itertools import product
from sklearn.cluster import Birch

def clasterization_plot(pca, true_classes, base_classes, elbow_classes, kelbow_visualizer, suptitle):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(24, 8))
    fig.suptitle(suptitle)
    ax[0].scatter(pca[:, 0], pca[:, 1], c=true_classes, cmap='viridis')
    ax[0].set_title("Без кластеризации")
    ax[0].set_xlabel("Признак 1")
    ax[0].set_ylabel("Признак 2")
    ax[1].scatter(pca[:, 0], pca[:, 1], c=base_classes, cmap='viridis')
    ax[1].set_title("Кластеризация c 2 классами")
    ax[1].set_xlabel("Признак 1")
    ax[1].set_ylabel("Признак 2")
    kelbow_visualizer.set_title("Метод локтя")
    kelbow_visualizer.ax.set_xlabel("Число кластеров")
    kelbow_visualizer.ax.set_ylabel("Score")
    ax[2].scatter(pca[:, 0], pca[:, 1], c=elbow_classes, cmap='viridis')
    ax[2].set_title("Кластеризация c " + str(kelbow_visualizer.elbow_value_) + " классами")
    ax[2].set_xlabel("Признак 1")
    ax[2].set_ylabel("Признак 2")

def clasterization_metrics(X_train, y_train, base_classes, elbow_classes, suptitle):
    print("Метрики оценки кластеризации", suptitle, "с базовым числом кластеров и с числом кластеров подобранным «правилом локтя»")
    print('Adjusted Rand Index: {:.4f}, {:.4f}'.format(adjusted_rand_score(y_train, base_classes), adjusted_rand_score(y_train, elbow_classes)))
    print('Adjusted Mutual Information: {:.4f}, {:.4f}'.format(adjusted_mutual_info_score(y_train, base_classes),
          adjusted_mutual_info_score(y_train, elbow_classes)))
    print('Homogeneity: {:.4f}, {:.4f}'.format(homogeneity_score(y_train, base_classes),
          homogeneity_score(y_train, elbow_classes)))
    print('Completeness: {:.4f}, {:.4f}'.format(completeness_score(y_train, base_classes),
          completeness_score(y_train, elbow_classes)))
    print('V-Measure: {:.4f}, {:.4f}'.format(v_measure_score(y_train, base_classes),
          v_measure_score(y_train, elbow_classes)))
    print('Silhouette: {:.4f}, {:.4f}'.format(silhouette_score(X_train, base_classes),
          silhouette_score(X_train, elbow_classes)))

def birch_grid_search(parameters, X_train, y_train):
    best_score = -2
    for params in product(*parameters.values()):
        clustering = Birch(**dict(zip(parameters.keys(), params))).fit(X_train)
        score = adjusted_rand_score(y_train, clustering.labels_)
        if score > best_score:
            best_score = score
            best_model = clustering
            best_params = dict(zip(parameters.keys(), params))
    print("Лучшие параметры", best_params)
    return best_model
