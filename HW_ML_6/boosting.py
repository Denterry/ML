from __future__ import annotations

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:


    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):
        # bootstrap_sample = np.random.choice(x, size=x.shape[0], reaplce=True)
        
        # Generate bootstrap sample for our base_model
        idx = np.random.choice(y.shape[0], size=int(self.subsample * y.shape[0]), replace=True)
        bootstrap_sample_x = x[idx]
        bootstrap_sample_y = y[idx]

        regressor = None
        if self.base_model_class:
            regressor = self.base_model_class(**self.base_model_params)
        else:
            regressor = self.base_model_class()

        regressor.fit(bootstrap_sample_x, bootstrap_sample_y)
        new_predictions = regressor.predict(x)

        gamma = self.find_optimal_gamma(y=y, old_predictions=predictions, new_predictions=new_predictions)
        predictions += self.learning_rate * gamma * new_predictions
        
        self.gammas.append(gamma)
        self.models.append(regressor)

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])

        for _ in range(self.n_estimators):
            self.fit_new_base_model(x_train, y_train, train_predictions)

            train_predictions = self.predict_proba(x_train)[:, 1];
            valid_predictions = self.predict_proba(x_valid)[:, 1];

            trains_loss_fn = self.loss_fn(y=y_train, z=train_predictions)
            valid_loss_fn = self.loss_fn(y=y_valid, z=valid_predictions)

            self.history['train_loss'].append(trains_loss_fn);
            self.history['valid_loss'].append(valid_loss_fn);
            
            if self.early_stopping_rounds is not None:
                if (len(self.history['valid_loss']) > self.early_stopping_rounds and
                    valid_loss_fn > np.min(self.history['valid_loss'][-self.early_stopping_rounds:])):
                    break;

        if self.plot:
            plt.plot(self.history['train_loss'], label='Results of loss on Train Sample')
            plt.plot(self.history['valid_loss'], label='Results of loss on Valid Sample')
            plt.xlabel('Models')
            plt.ylabel('Steps')
            plt.show()

    def predict_proba(self, x):
        predictions = np.zeros((x.shape[0], 2))
        for gamma, model in zip(self.gammas, self.models):
            predictions[:, 1] += self.learning_rate * gamma * model.predict(x)
        
        predictions[:, 0] = 1 - predictions[:, 1]
        return self.sigmoid(predictions)

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    def feature_importances_(self):
        feature_importances = np.zeros_like(self.models[-1].feature_importances_)

        for model in self.models:
            np.add(feature_importances, model.feature_importances_, out=feature_importances, casting="unsafe")

        feature_importances /= len(self.models)
        feature_importances /= np.sum(feature_importances)
        return feature_importances