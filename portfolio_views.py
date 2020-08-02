import numpy as np
import scipy as sc
import pandas as pd
import openpyxl as ox
from scipy.optimize import minimize
import enum
from collections.abc import Iterable
import copy

class Portfolio():

    def __init__(self, data, cov=None, kappa=None):
        """
        Create a portoflio object with assets allocation, returns, covariance matrix.

        Args:
            data (dict or pd.DataFrame): dictionary containing assert names as keys or dataframe containg asset names as columns, default weights are equal weights, default returns are zeros
            cov (dict or pd.DataFrame): covariane dataframe, default is no correlation (identity matrix)
            kappa (float): Risk reversion parameter
        """

        self.df = data
        self.cov = cov
        self.kappa = 1

    @property
    def df(self):
        return self._df

    def display(self):
        return self._df.style.format('{:,.2%}')

    @df.setter
    def df(self, data):
        if not (isinstance(data, dict) or isinstance(data, pd.DataFrame)):
            raise ValueError(
                "Invalid type: data must be either dataframe or dict with name of assets as keys")

        if isinstance(data, dict):
            self._df = pd.DataFrame(data)

        if 'r' not in self._df.index:
            if 'r' in self._df.columns:
                self._df = self._df.transpose()
            else:
                self._df['r'] = 0.

        if 'w' not in self._df.index:
            if 'w' in self._df.columns:
                self._df = self._df.transpose()
            else:
                self._df.loc['w', :] = 1. / self._df.shape[1]

        self._df.sort_index(axis=0, inplace=True)
        self._df.sort_index(axis=1, inplace=True)

    @property
    def cov(self):
        return self._cov

    @cov.setter
    def cov(self, cov):
        if cov is not None:
            if not isinstance(cov, pd.DataFrame):
                raise ValueError(
                    "Covariance dataframe {} must be symmetric with same index and columns names".format(cov))
            if (cov.index != cov.columns).all():
                raise ValueError(
                    "Covariance indices: {} must be equal to covariance columns: {}").format(cov.index, cov.columns)
            if not cov.equals(cov.transpose()):
                raise ValueError(
                    "Covariance dataframe {} must be symmetric".format(cov))
            self._cov = cov

        else:
            print("Assuming no correlation between assets, covariance matrix = I")
            self._cov = pd.DataFrame(np.eye(self._df.shape[1]),
                                     columns=self.df.columns,
                                     index=self.df.index)

        self._cov.sort_index(axis=0, inplace=True)
        self._cov.sort_index(axis=1, inplace=True)

    @property
    def w(self):
        return self._df.loc['w', :]

    @property
    def r(self):
        return self._df.loc['r', :]

    def optim_w(self, inplace=True):
        """
        Calcule les poids optimaux avec une matrice de covariance et des sur-rendements

        Args:
            inplace (bool, optional): override les poids dans l'instance si Vrai (default), output poids (np.array) sinon
        """
        w = np.linalg.inv(self._cov.values) @ self.r.values
     # self.kappa = w.sum()
        if inplace:
            self._df.loc['w', :] = w
        else:
            return w

    # Calcule les sur-rendements implicites pour une matrice de covariance et des poids normalisés
    def imp_ret(self, inplace=True):
        """
        Calcule les sur-rendements implicites pour une matrice de covariance et des poids normalisés

        Args:
            inplace (bool, optional): override les rendements dans l'instance si Vrai (default), output r (np.array) sinon
        """
        if inplace:
            self._df.loc['r', :] = self.kappa * \
                np.dot(self._cov.values, self.w.values)
        else:
            return self.kappa * np.dot(self._cov.values, self.w.values)

    def imp_kap(self, inplace=True):
        """"
        Calcule l'aversion au risque explicite
        """
        if inplace:
            self.kappa = self.r.iloc[0] / self._cov.iloc[0, :].dot(self.w)
        else:
            return self.r.iloc[0] / self._cov.iloc[0, :].dot(self.w)

    # Calcule le rendement attendu de Black-litterman
    # tau est le scalaire à calibrer
    # P la matrice K*N qui associe les K views aux N acfifs
    # Q la matrice des rendements sur les vues (relatifs ou absolus)
    # Ome la matrice de covariance sur l'incertitude des vues

    def post_ret(self, tau, P, Q, Ome):
        InvSig = np.linalg.inv(tau * self.cov)
        InvOme = np.linalg.inv(Ome)
        X = InvSig + np.dot(np.dot(P.t, InvOme), P)
        Y = np.dot(InvSig, self.r) + np.dot(np.dot(P.t, InvOme), Q)
        return np.linalg.solve(X, Y)

    # Calcule le rendement attendu de Black-litterman avec des vues ayant 100% de certitude
    # tau est le scalaire à calibrer
    # V l'objet View qui contient :
    # P la matrice K*N qui associe les K views aux N acfifs
    # Q la matrice des rendements sur les vues (relatifs ou absolus)

    def post_ret100(self, V, tau=1):
        X = V.Q - np.dot(V.P, self.r).reshape(len(V.Q), 1)
        PSPt = np.linalg.inv(np.dot(np.dot(V.P, 1 * self.cov), V.P.T))
        Z = self.r + np.dot(tau * np.dot(self.cov, V.P.T),
                            np.dot(PSPt, X)).reshape(V.P.shape[1])
        copy = self.copy()
        copy.r = Z
        copy.optim_w()
        return copy.r

    # Calcule le rendement pour la k-ème vue avec une certitude de 100%
    def post_ret100_k(self, V, k, tau=1):
        cop = self.copy()
        V_k = Views(cop)
        V_k.P = V.P[k].reshape(1, v.P.shape[1])
        V_k.Q = V.Q[k].reshape(1, 1)
        V_k.conf = V.conf[k]
        return cop.post_ret100(V_k, tau)

    def w_pk(self, V, k):
        w = self.post_ret100_k(V, k)
        return w + (w - self.w) * V.conf

    @staticmethod
    def f_k(omega, P, V, k, tau=1):
        InvSig = np.linalg.inv(tau * P.cov)
        first_term = np.linalg.inv(InvSig + np.outer(V.P[k], V.P[k]) / omega)
        second_term = np.dot(InvSig, P.r) - V.P[k] * V.Q[k] / omega
        w_k = np.linalg.inv(P.kappa * P.cov).dot(first_term.dot(second_term))
        return np.linalg.norm(P.w_pk(V, k) - w_k)


# Vue avec prise en compte de la confiance

class Views:

    def __init__(self, views={}):
        self._df = pd.DataFrame()
        self.view_list = self._df.to_dict("records")

    def add_views(self, df):
        """Given a dataframe of view(s), add these views to the Views object. The dataframe must contain
        the name of the assets as columns as well as an 'r' column for the expected returns.

        Args:
            df (pandas dataframe): Contain asset names as columns for coefficients,
                                   r column for expected return
                                   and c column for confidence (default 100% confidence)

        Raises:
            IndexError: if 'r' not in column
            ValueError: For empty dataframe
        """
        if df.shape[1] < 2:
            raise ValueError(""" Please provide valid view dataframe.
                                 The dataframe must contain the name of the assets as columns,
                                 an 'r' column for the expected returns,
                                 an optional 'c' column for confidence coefficients""")

        if 'r' not in df.columns:
            raise IndexError(" 'r' for returns must be present in DataFrame")

        if 'c' not in df.columns:
            # Assume 100% confience if 'c' column does not exist
            print("Warning: assuming 100% confidence for view(s)")
            df['c'] = 1.

        # add views to existing ones
        self._df = self._df.append(df)

        # Remove duplicates
        df_dup = self._df[self._df.duplicated()]
        if len(df_dup) > 0:
            print("Following view(s) already exist:")
            for coeff, r, c in zip(df_dup.drop(['r', 'c'], axis=1).to_dict('records'), df_dup['r'], df_dup['c']):
                print(Views.view_to_str(coeff, r, c))
            print("Dropping duplicates")
            self._df = self._df.drop_duplicates()

        self._df.sort_index(axis=0, inplace=True)
        self._df.sort_index(axis=1, inplace=True)
        self._df = self._df.fillna(0.)

    def remove_views(self, names_or_indices):
        return print("NOT IMPLEMENTED")

    @property
    def df(self):
        return self._df

    @property
    def P(self):
        df_coefficients = self._df.drop(['r', 'c'], axis=1)
        return df_coefficients

    def __getitem__(self, i):
        return {'P': self._df.drop(['r', 'c'], axis=1).loc[i, :],
                'r': self._df.loc[i, 'r'],
                'c': self._df.loc[i, 'c']}

    def check_if_view_already_exist(self, view):
        """
        Given a Views instance and a view dictionnary, check if that view is already expressed in the instance

        Args:
            view (dict): contain asset names as keys, with a 'r' key for returns and 'c' key for confidence

        Returns:
            bool: True if view already exists, False otherwise
        """
        if view in self.df.to_dict('records'):
            return True
        else:
            return False

    @staticmethod
    def view_to_str(coeff_dict, r, c):
        """
        Return the view in human readable str given a coefficient dictionary (keys are asset names and values
        are float coefficients), a return iterable and a confidence iterable.

        Args:
            coeff_dict (dict): coefficient values keyed by asset name
            r (iter): iterable of returns (float)
            c (iter): iterable of confidence (float)

        Returns:
            string: Human readable view
        """
        s = " ".join(['{:.1%} {}'.format(coeff, name)
                      for (name, coeff) in zip(coeff_dict.keys(), coeff_dict.values())])
        s += " with expected return {:.1%} (confidence: {:.1%})".format(r, c)
        return s

    def __repr__(self):
        """
        Default display of a Views object
        """
        coeff_df = self._df.drop(['r', 'c'], axis=1)
        phrase = "Views object with: \n"
        phrase += "\n".join([Views.view_to_str(coeff_dict,
                                               r,
                                               c) for coeff_dict, r, c in zip(coeff_df.to_dict('record'),
                                                                              self.df["r"],
                                                                              self.df["c"])

                             ])
        return phrase


class PortfolioProblem:

    def __init__(self, portfolio, views):
        """
        param Portfolio portfolio: portfolio containing names, returns, covariance and kappa
        param Views views: View object with P, Q matrices and confidence

        """
        self.portfolio = portfolio
        self.views = views

    def post_ret100_k(self, k, tau=1.0):
        """
        Calcule le rendement attendu de Black-litterman avec des vues ayant 100% de certitude

        Args:
            k (int): view index
            tau (float, optional): tau est le scalaire à calibrer

        Returns:
            TYPE: Description
        """

        # Qk - pk Pi
        X = self.views[k]['r'] - self.views[k]['P'].dot(self.portfolio.r)
        # inv(pk tau Sigma pk')
        PkPiPk_1 = 1. / \
            self.views[k]['P'].dot(
                tau * self.portfolio.cov.dot(self.views[k]['P']))
        r_100 = self.portfolio.r + tau * \
            self.portfolio.cov.dot(self.views[k]['P']) * PkPiPk_1 * X
        new_portfolio = self.copy()
        copy.r = Z
        copy.optim_w()
        return copy.r

    def post_ret100(self, tau=1.0):
        """
        Calcule le rendement attendu de Black-litterman avec des vues ayant 100% de certitude

        Args:
            tau (float, optional): tau est le scalaire à calibrer

        Returns:
            TYPE: Description
        """
        X = self.views.df['r'] - np.dot(V.P, self.r).reshape(len(V.Q), 1)
        PSPt = np.linalg.inv(np.dot(np.dot(V.P, 1 * self.cov), V.P.T))
        Z = self.r + np.dot(tau * np.dot(self.cov, V.P.T),
                            np.dot(PSPt, X)).reshape(V.P.shape[1])
        copy = self.copy()
        copy.r = Z
        copy.optim_w()
        return copy.r

    @staticmethod
    def f_k(omega, P, V, k, tau=1):
        """
        param float omega: diagonal element of covariance matrix of error term
        param Portfolio P:
        param Views V:
        param int k:


        return square difference of new weights vs target weights
        """
        InvSig = np.linalg.inv(tau * P.cov)
        first_term = np.linalg.inv(InvSig + np.outer(V.P[k], V.P[k]) / omega)
        second_term = np.dot(InvSig, P.r) - V.P[k] * V.Q[k] / omega
        w_k = np.linalg.inv(P.kappa * P.cov).dot(first_term.dot(second_term))
        return np.linalg.norm(P.w_pk(V, k) - w_k)


# Vue basique, sans prise en compte de confiance

class ViewsA:
    def __init__(self, Port):
        self.P = None
        self.Q = None
        self.Ome = None
        self.Port = Port

    # ajoute une vue
    def add(self, name, w_v, r_v):
        X = np.zeros((1, len(self.Port.assets)))
        i = 0
        for n in name:
            X[0, self.Port.assets.index(n)] = w_v[i]
            i = i + 1

        if self.P is None:
            self.P = X
        else:
            self.P = np.vstack((self.P, X))

        if self.Q is None:
            self.Q = np.array([[r_v]])
        else:
            self.Q = np.vstack((self.Q, np.array([[r_v]])))

        omega = np.dot(X, np.dot(self.Port.cov, X.T))

        if self.Ome is None:
            self.Ome = omega
        else:
            self.Ome = np.block([[self.Ome, np.zeros((np.shape(self.Ome)[0], 1))], [
                np.zeros((1, np.shape(self.Ome)[0])), omega]])


if __name__ == "__main__":

    d = {
        'equities': dict(r=0.07),
        'fixed_income': dict(r=0.03),
    }

    port2 = Portfolio(d,
                      cov=pd.DataFrame(
                          np.array([[1.2, -0.1], [-0.1, 0.3]]), columns=d.keys(), index=d.keys()),
                      kappa=0.1)
    v = Views()
    df = pd.DataFrame(
        {
            'fixed_income': [0, 1, 1],
            'equities': [1, 0, 0],
            'r': [0.05, 0.01, 0.01],
            'c': [0.5, 0.5, 0.5]
        })

    v.add_views(df)
