import numpy as np
import scipy as sc
import pandas as pd
import openpyxl as ox
from scipy.optimize import minimize, fmin_bfgs
import enum
from collections.abc import Iterable
from collections import namedtuple
import copy
# logging.basicConfig(filename=__name__, level=logging.DEBUG)
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Portfolio():

    def __init__(self, data=None, cov=None, kappa=None):
        """
        Create a portoflio object with assets allocation, returns, covariance matrix.

        Args:
            data (dict or pd.DataFrame): dictionary containing assert names as keys or dataframe containg asset names as columns, default weights are equal weights, default returns are zeros
            cov (dict or pd.DataFrame): covariane dataframe, default is no correlation (identity matrix)
            kappa (float): Risk reversion parameter
        """
        if data is not None:
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
            logging.error(
                "Invalid type: data must be either dataframe or dict with name of assets as keys")

        if isinstance(data, dict):
            self._df = pd.DataFrame(data)
        else:
            self._df = data

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
                logging.error(
                    "Covariance dataframe {} must be symmetric with same index and columns names".format(cov))
            if (cov.index != cov.columns).all():
                logging.error(
                    "Covariance indices: {} must be equal to covariance columns: {}").format(cov.index, cov.columns)
            if not cov.equals(cov.transpose()):
                logging.error(
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

    def read_xlsx_ptf(self, path, r_name='r', w_name='w', sheet_name=0):
        df = pd.read_excel(path, sheet_name=sheet_name, index_col=0)
        df = df.rename(index={r_name: 'r', 
                                w_name:'w'})
        self.df = df

    def read_xlsx_cov(self, path, sheet_name=1):
        df = pd.read_excel(path, sheet_name=sheet_name, index_col=0)
        self.cov = df

    @r.setter
    def r(self, r):
        self._df.loc['r', :] = r

    def optim_w(self, inplace=True):
        """
        Calcule les poids optimaux avec une matrice de covariance et des sur-rendements

        Args:
            inplace (bool, optional): override les poids dans l'instance si Vrai (default), output poids (np.array) sinon
        """
        w = np.linalg.inv(self.kappa * self._cov.values) @ self.r.values
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

    def w_pk(self, V, k):
        w = self.post_ret100_k(V, k)
        return w + (w - self.w) * V.conf

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)

    @staticmethod
    def f_k(omega, P, V, k, tau=1):
        InvSig = np.linalg.inv(tau * P.cov)
        first_term = np.linalg.inv(InvSig + np.outer(V.P[k], V.P[k]) / omega)
        second_term = np.dot(InvSig, P.r) - V.P[k] * V.Q[k] / omega
        w_k = np.linalg.inv(P.kappa * P.cov).dot(first_term.dot(second_term))
        return np.linalg.norm(P.w_pk(V, k) - w_k)


# single view representation
class View(namedtuple('View', 'P r c')):
    def __repr__(self):
        return Views.view_to_str(self.P.to_dict(), self.r, self.c)

# Multiple views class
class Views:

    def __init__(self, views={}):
        # default views in dataframe form
        self._df = pd.DataFrame()

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
            logging.info("Warning: assuming 100% confidence for view(s)")
            df['c'] = 1.

        # add views to existing ones
        self._df = self._df.append(df)

        # Remove duplicates
        df_dup = self._df[self._df.duplicated()]
        if len(df_dup) > 0:
            logging.info("Following view(s) already exist:")
            for coeff, r, c in zip(df_dup.drop(['r', 'c'], axis=1).to_dict('records'), df_dup['r'], df_dup['c']):
                print(Views.view_to_str(coeff, r, c))
            logging.info("Dropping duplicates")
            self._df = self._df.drop_duplicates()

        self._df.reset_index(drop=True, inplace=True)
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
        return View(P=self._df.drop(['r', 'c'], axis=1).loc[i, :],
                    r=self._df.loc[i, 'r'],
                    c=self._df.loc[i, 'c'])

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

        phrase = "Views object with: \n"
        for view in self:
            phrase += str(view) + "\n"

        return phrase

    def to_list(self):
        return [self[k] for k in range(self._df.shape[0])]

    def __len__(self):
        return self._df.shape[0]

    def __iter__(self):
        return (self[k] for k in range(self._df.shape[0]))

    def read_xlsx_views(self, path, r_name='r', c_name='c', sheet_name=2):
        df = pd.read_excel(path, sheet_name=sheet_name)
        df = df.rename(columns={r_name: 'r', 
                                c_name: 'c'})
        self.add_views(df)



class PortfolioProblem:

    def __init__(self, portfolio, views):
        """
        param Portfolio portfolio: portfolio containing names, returns, covariance and kappa
        param Views views: View object with P, Q matrices and confidence

        """
        self.portfolio = portfolio
        self.views = views

    def post_ret100_k(self, view_k, inplace=True, tau=1.0):
        """
        Calcule le rendement attendu de Black-litterman avec des vues ayant 100% de certitude

        Args:
            view_k (View object): single view to take in consideration
            tau (float, optional): tau est le scalaire à calibrer

        Returns:
            TYPE: Portfolio if inplace=False
        """

        # Qk - pk Pi
        X = view_k.r - view_k.P.dot(self.portfolio.r)
        # inv(pk tau Sigma pk')
        PkPiPk_1 = 1. / view_k.P.dot(
            tau * self.portfolio.cov.dot(view_k.P))
        r_100 = self.portfolio.r + tau * \
            self.portfolio.cov.dot(view_k.P) * PkPiPk_1 * X
        if inplace:
            # new_portfolio is not a copy here but is just another name for self.portfolio
            new_portfolio = self.portfolio
        else:
            new_portfolio = self.portfolio.deepcopy()
        new_portfolio.r = r_100
        # optimize weights wrt new expected returns, covariance not changed for r distribution
        new_portfolio.optim_w()
        if not inplace:
            return new_portfolio

    def w_pk(self, view_k):
        """
        Calculate the w_%k, linear average between current weight in self.portfolio and weight
        using view k with 100% confidence

        Args:
            view_k (View object): single view to take in consideration

        Deleted Parameters:
            inplace (bool, optional): Not sure if we need that
        """
        dummy_portfolio = self.post_ret100_k(view_k, inplace=False)
        confidence_k = view_k.c
        w_pk = self.portfolio.w + \
            (dummy_portfolio.w - self.portfolio.w) * confidence_k
        return w_pk

    def post_ret100(self, inplace=True, tau=1.0):
        """
        Calcule le rendement attendu de Black-litterman avec des vues ayant 100% de certitude

        Args:
            tau (float, optional): tau est le scalaire à calibrer

        Returns:
            TYPE: Description
        """
        print("NOT IMPLEMENTED")

    @staticmethod
    def f_k(omega, pproblem, view_k, tau=1):
        """
        param float omega: diagonal element of covariance matrix of error term
        param Portfolio P:
        param Views V:
        param int k:


        return square difference of new weights vs target weights
        """

        # Il faut probablement modifier le Sigma posterior

        w_pk = pproblem.w_pk(view_k)
        InvSig = np.linalg.inv(tau * pproblem.portfolio.cov)
        first_term = np.linalg.inv(
            InvSig + np.outer(view_k.P, view_k.P) / omega)
        second_term = np.dot(InvSig, pproblem.portfolio.r) + \
            view_k.P * view_k.r / omega
        new_cov = pproblem.portfolio.cov  # c'est le sigma "star" ?
        w_k = np.linalg.solve(
            pproblem.portfolio.kappa * new_cov, np.dot(first_term, second_term))
        return np.linalg.norm(w_pk - w_k)

    @staticmethod
    def f(omegas, pproblem, tau=1):
        """
        returns list of difference in norm between w_pk and w_k, given a portfolio problem and list of omegas

        Args:
            omegas (iterable):
            pproblem (PortfolioProblem object):
            tau (int, optional):

        Returns:
            TYPE: list of outputs of f_k given the portfolio problem
        """
        return np.linalg.norm([PortfolioProblem.f_k(omega, pproblem, view, tau=tau) for omega, view in zip(omegas, pproblem.views)])

    def compute_Omega(self, tau=1.0):
        """
        Calculate optimal omegas given the portfolio problem using Nelder-Mead optimization

        Args:
            tau (float, optional): Description

        Returns:
            np.array (2D): Omega matrix
        """
        # initial guesses
        omegas_0 = np.ones(len(self.views)) * 0.1
        logging.info("""Setting up optimization for portfolio:
            {}
            and views:
            {}""".format(self.portfolio.df, self.views))
        result = minimize(PortfolioProblem.f, omegas_0,
                          args=(self, tau), method='Nelder-Mead')
        if result.success:
            logging.info(result.message)
            return result.x
        else:
            logging.error(result.message + "{}".format(result))

    def compute_Omega_analytical(self, tau=1.0):
        """
        Calculate optmal omegas using analytical method (Walters et al., 2007)

        Args:
            tau (float, optional): Description

        Returns:
            np.array (2D): Omega matrix
        """
        if all(self.views.df.c.values < 0) or all(self.views.df.c.values > 1):
            logging.error(
                "Confidence in views: {} must be strictly between 0 and 1".format(self.views.df.c.values))
        alphas = np.array([(1.0 - view.c) / (view.c) for view in self.views])
        omegas_star = np.array([alpha * (view.P @ self.portfolio.cov).dot(view.P)
                                for alpha, view in zip(alphas, self.views)])
        return omegas_star

    def compute_new_returns(self, omegas):
        """

        Args:
            omegas (TYPE): Description
        """
        if any(omegas == 0):
            logging.warning(
                "omegas must be strictly different than zero: {}".format(omegas))
        sigma_inv = np.linalg.inv(self.portfolio.cov)
        try:
            omega_inv = np.diag(1. / omegas)
        except (AssertionError, TypeError) as e:
            logging.error(e)
            raise e
        first_term = np.linalg.inv(
            sigma_inv + np.dot(self.views.P.T, omega_inv).dot(self.views.P))
        second_term = np.dot(sigma_inv, self.portfolio.r) + \
            np.dot(self.views.P.T, omega_inv).dot(self.views.df.r)

        return first_term @ second_term

    def post_portfolio(self, omega_analytical=True, tau=1.0):
        """
        Return the new portfolio taking into account the views of the portfolio problem

        Args:
            omega_analytical (bool, optional): Use analytical (Default) or optimization method
            tau (float, optional):

        Returns:
            portfolio object: new porfolio
        """
        if omega_analytical:
            omegas = self.compute_Omega_analytical(tau)
        else:
            omegas = self.compute_Omega(tau)
        E_r = self.compute_new_returns(omegas)
        new_portfolio = copy.deepcopy(self.portfolio)
        new_portfolio.r = E_r
        new_portfolio.optim_w()
        return new_portfolio
