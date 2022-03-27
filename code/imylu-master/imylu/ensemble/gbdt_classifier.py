# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-07-05 17:37:34
@Last Modified by: tushushu
@Last Modified time: 2018-07-05 17:37:34
"""
from typing import List
import numpy as np
from numpy import ndarray

from ..utils.utils import sigmoid
from .gbdt_base import GradientBoostingBase, RegressionTree


class GradientBoostingClassifier(GradientBoostingBase):
    """Gradient Boosting Classifier"""

    def _get_init_val(self, label: ndarray):
        """Calculate the initial prediction of y
        Estimation function (Maximize the likelihood):
        z = fm(xi)
        p = 1 / (1 + e**(-z))

        Likelihood function, yi <- y, and p is a constant:
        Likelihood = Product(p^yi * (1-p)^(1-yi))

        Loss function:
        L = Sum(yi * Logp + (1-yi) * Log(1-p))

        Get derivative of p:
        dL / dp = Sum(yi/p - (1-yi)/(1-p))

        dp / dz = p * (1 - p)

        dL / dz = dL / dp * dp / dz
        dL / dz = Sum(yi * (1-p) - (1-yi)* p)
        dL / dz = Sum(yi) - Sum(1) * p

        Let derivative equals to zero, then we get initial constant value
        to maximize Likelihood:
        p = Mean(yi)
        1 / (1 + e**(-z)) = Mean(yi)
        z = Log(Sum(yi) / Sum(1-yi))
        ----------------------------------------------------------------------------------------

        Arguments:
            y {list} -- 1d list object with int or float

        Returns:
            float
        """

        n_rows = len(label)
        tot = label.sum()

        return np.log(tot / (n_rows - tot))

    @staticmethod
    def _get_score(idxs: List[int], prediction: ndarray, residuals: ndarray) -> float:
        """Calculate the regression tree leaf node value
        Estimation function (Maximize the likelihood):
        z = Fm(xi) = Fm-1(xi) + fm(xi)
        p = 1 / (1 + e**(-z))

        Likelihood function, yi <- y, and p is a constant:
        Likelihood = Product(p^yi * (1-p)^(1-yi))

        Loss Function:
        Loss(yi, Fm(xi)) = Sum(yi * Logp + (1-y) * Log(1-p))

        Taylor 1st:
        f(x + x_delta) = f(x) + f'(x) * x_delta
        f(x) = g'(x)
        g'(x + x_delta) = g'(x) + g"(x) * x_delta


        1st derivative:
        Loss'(yi, Fm(xi)) = Sum(yi - p)

        2nd derivative:
        Loss"(yi, Fm(xi)) = Sum((p - 1) * p)

        So,
        Loss'(yi, Fm(xi)) = Loss'(yi, Fm-1(xi) + fm(xi))
        = Loss'(yi, Fm-1(xi)) + Loss"(yi, Fm-1(xi)) *  fm(xi) = 0
        fm(xi) = - Loss'(yi, Fm-1(xi)) / Loss"(yi, Fm-1(xi))
        fm(xi) = Sum(yi - p) / Sum((1 - p) * p)
        fm(xi) = Sum(residual_i) / Sum((1 - p) * p)
        ----------------------------------------------------------------------------------------

        Arguments:
            idxs{List[int]} -- Indexes belongs to a leaf node.
            prediction {ndarray} -- Prediction of label.
            residuals {ndarray}

        Returns:
            float
        """

        numerator = residuals[idxs].sum()
        denominator = (prediction[idxs] * (1 - prediction[idxs])).sum()

        return numerator / denominator

    def _update_score(self, tree: RegressionTree, data: ndarray,
                      prediction: ndarray, residuals: ndarray):
        """update the score of regression tree leaf node.

        Arguments:
            tree {RegressionTree}
            data {ndarray} -- Training data.
            prediction {ndarray} -- Prediction of label.
            residuals {ndarray}
        """

        nodes = self._get_leaves(tree)

        regions = self._divide_regions(tree, nodes, data)
        for node, idxs in regions.items():
            node.avg = self._get_score(idxs, prediction, residuals)
        tree.get_rules()

    def predict_one_prob(self, row: ndarray) -> float:
        """Auxiliary function of predict_prob.

        Arguments:
            row {ndarray} -- A sample of testing data.

        Returns:
            float -- Prediction of label.
        """

        return sigmoid(self.predict_one(row))

    def predict_prob(self, data: ndarray) -> ndarray:
        """Get the probability of label.

        Arguments:
            data {ndarray} -- Testing data.

        Returns:
            ndarray -- Probabilities of label.
        """

        return np.apply_along_axis(self.predict_one_prob, axis=1, arr=data)

    def predict(self, data: ndarray, threshold=0.5) -> ndarray:
        """Get the prediction of label.

        Arguments:
            data {ndarray} -- Testing data.

        Keyword Arguments:
            threshold {float} -- (default: {0.5})

        Returns:
            ndarray -- Prediction of label.
        """

        prob = self.predict_prob(data)
        return (prob >= threshold).astype(int)
