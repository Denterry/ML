import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    feature_vector = np.array(feature_vector)
    target_vector = np.array(target_vector)

    sort_indexes = np.argsort(feature_vector)
    feature_vector = feature_vector[sort_indexes]
    target_vector = target_vector[sort_indexes]

    unique_features, unique_indxs = np.unique(feature_vector, return_index=True)
    thresholds = (unique_features[:-1] + unique_features[1:])/2
    ginies = []
    threshold_best = None
    gini_best = None

    cum_target = np.cumsum(target_vector)
    cum_sum_target = np.cumsum(target_vector)[:-1]
    cum_rev_sum_target = np.cumsum(target_vector[:0:-1])[::-1]

    p1_left =  cum_sum_target / (np.arange(1, len(feature_vector)))
    p0_left = 1.0 - p1_left
    p1_right =  (cum_target[-1] - cum_target)[:-1]  / (len(feature_vector) - np.arange(1, len(feature_vector)))
    p0_right = 1.0 - p1_right

    gini_left = 1.0 - p1_left**2 - p0_left**2
    gini_left = gini_left[unique_indxs - 1][1:]
    gini_right = 1.0 - p1_right**2 - p0_right**2
    gini_right = gini_right[unique_indxs - 1][1:]

    ginies = ((-1)*((np.arange(1, len(feature_vector))[unique_indxs - 1][1:] / len(feature_vector)) * gini_left) + (-1)*(((len(feature_vector) - np.arange(1, len(feature_vector)))[unique_indxs - 1][1:] / len(feature_vector)) * gini_right))

    best_index = np.argmax(ginies)
    threshold_best = thresholds[best_index]
    gini_best = np.max(ginies)

    return thresholds, ginies, threshold_best, gini_best

class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
    
    def get_params(self, deep=True):
        params = {
            'feature_types': self._feature_types,
            'max_depth': self._max_depth,
            'min_samples_split': self._min_samples_split,
            'min_samples_leaf': self._min_samples_leaf
        }
        return params

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        # if np.all(sub_y != sub_y[0]):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        # for feature in range(1, sub_X.shape[1]):
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    # ratio[key] = current_count / current_click
                    ratio[key] = current_click / current_count
                # sorted_categories = list(map(lambda x: x[1], sorted(ratio.items(), key=lambda x: x[1])))
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            if np.all(feature_vector == feature_vector[0]):
                continue
            
            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                            filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None or depth == self._max_depth or \
            (self._min_samples_split is not None and len(sub_y) <= self._min_samples_split) or \
            (self._min_samples_leaf is not None and (len(sub_y[split]) <= self._min_samples_leaf or len(sub_y[np.logical_not(split)]) <= self._min_samples_leaf)):
            node["type"] = "terminal"
            # node["class"] = Counter(sub_y).most_common(1)
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth+1)
        # self._fit_node(sub_X[np.logical_not(split)], sub_y[split], node["right_child"], depth + 1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth+1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
        else:
            feature_type = self._feature_types[node["feature_split"]]
            if feature_type == "real":
                if x[node["feature_split"]] < node["threshold"]:
                    return self._predict_node(x, node["left_child"])
                else:
                    return self._predict_node(x, node["right_child"])
            elif feature_type == "categorical":
                if x[node["feature_split"]] in node["categories_split"]:
                    return self._predict_node(x, node["left_child"])
                else:
                    return self._predict_node(x, node["right_child"])
            else:
                raise ValueError("Invalid feature type")

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

