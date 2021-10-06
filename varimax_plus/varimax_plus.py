# External modules
import numpy as np
from scipy.stats import pearsonr, ttest_1samp


class VarimaxPlus:
    """
    Variation of Varimax that uses a bootstrap test to determine if the found weight is significant
    """

    def __init__(self, data, truncate_by='max_comps', max_comps=60, fraction_explained_variance=0.9,
                 boot_axis: int = 0, boot_rep: int = 100, boot_samples: int = 100, alpha_level: float = 0.05,
                 verbose=True):

        self.data = data  # Shape L x T -> L x max_comps

        # Varimax
        self.truncate_by = truncate_by
        self.max_comps = max_comps
        self.fraction_explained_variance = fraction_explained_variance

        # Bootstrap
        self.boot_axis = boot_axis
        self.boot_rep = boot_rep
        self.boot_samples = boot_samples

        # Others
        self.alpha_level = alpha_level
        self.verbose = verbose

        # Empty attributes
        self.results = None
        self.weights = None
        self.boot_data = None
        self.var_results = None
        self.boot_results = None
        self.mask_components = None
        self.threshold_weights = None

        # Addition for testing
        test1 = "We should pull and see what happens"

    def varimax_plus(self):
        """
        The main funciton, called when the class is called.
        :return:
        """
        # Checks
        # Check if data has the correct shape

        # For redability
        varimax_dict = {
            "truncate_by": self.truncate_by,
            "max_comps": self.max_comps,
            "fraction_explained_variance": self.fraction_explained_variance,
            "verbose": self.verbose

        }

        # Perform varimax once
        varimax = Varimax(data=self.data, **varimax_dict)
        var_results = varimax()
        self.var_results = var_results

        self.weights = var_results["weights"]

        # Do bootstrap on time.
        self.boot_data = self.butstrap(self.data, axis=self.boot_axis, n_repetitions=self.boot_rep,
                                       n_samples=self.boot_samples)

        # Perfom varimax on boostrap samples
        self.boot_results = [Varimax(d, **varimax_dict)() for d in self.boot_data]

        self.threshold_hypothesis()

    def threshold_hypothesis(self):

        # Correctly order all the weights using partial correlation
        boot_weights_sorted = [self.find_close_permutation(self.weights, s["weights"]) for s in self.boot_results]
        boot_weights_sorted = np.stack(boot_weights_sorted)

        # Do hypothesis testing for each grid-point of each component
        components_p_values = []
        for comp in range(boot_weights_sorted.shape[2]):  # Number of components
            components_p_values.append(ttest_1samp(boot_weights_sorted[:, :, comp], popmean=0).pvalue)

        mask_components = [(s < self.alpha_level) for s in components_p_values]
        mask_components = np.stack(mask_components, axis=1)
        self.mask_components = mask_components

        self.threshold_weights = self.weights * mask_components

        self.results = self.var_results
        self.results["old_weights"] = self.results["weights"]
        self.results["weights"] = self.threshold_weights

    @staticmethod
    def butstrap(data: np.ndarray, axis: int = 0, n_repetitions: int = 100,
                 n_samples: int = None) -> list:
        """
        Does generate n new repetitions of a given dataset bootrstaping on a given axis.
        :param data: Dataset, must be of ndim = 2
        :param axis: The axis in which the samples are taken.
        :param n_repetitions: Number of final datasets
        :param n_samples: amount of samples that we want in each bootstraped dataset, if None then the same original size
        :return: a list of repetitions of the bootstraped original dataset
        """


        # Checks
        if data.ndim != 2:
            raise ValueError("data must be of dimension 2")

        if axis not in [0, 1]:
            raise ValueError("axis must be 0 or 1")

        # Main variables
        if axis == 1:
            data = data.transpose()

        # Vars
        n_org_samples = data.shape[0]  # Original number of samples

        if n_samples is None:
            n_samples = n_org_samples

        # Bootstrap positions
        idx_samples = [np.random.choice(list(range(n_org_samples)), size=n_samples, replace=True)
                       for _ in range(n_repetitions)]

        results = [data[s, :] for s in idx_samples]

        if axis == 1:
            results = [x.transpose() for x in results]

        return results

    @staticmethod
    def find_close_permutation(original_data: np.ndarray, permuted_data: np.ndarray, axis: int = 1) -> np.ndarray:
        """
        Algorithm 2 of the SAVAR paper
        :param original_data: Data that has the correct axis orders
        :param permuted_data: Data that has the axis permuted
        :return: permuted data according to pearson coefficient of correlation of the original one
        """

        # Checks
        if original_data.ndim != 2 and permuted_data != 2:
            raise ValueError("data must be of dimension 2")

        if original_data.shape[axis] < permuted_data.shape[axis]:
            raise ValueError("Original Dataset must have at least the same number of dim in axis")

        if axis not in [0, 1]:
            raise ValueError("axis must be 0 or 1")

        if axis == 0:
            original_data.transpose()
            permuted_data.transpose()

        n_org_rows = original_data.shape[1]
        n_per_rows = permuted_data.shape[1]
        non_check_rows = list(range(n_per_rows))
        best_maches = []

        # For each row of original data
        # TODO: Use np.corrcoef() to improve efficiency
        for o_row in range(n_org_rows):  # For each original column
            pear_coeffs = []
            for p_row in non_check_rows:  # For each permuted column
                pear_coeffs.append(pearsonr(original_data[:, o_row], permuted_data[:, p_row])[0])

            best_idx = np.argmax(list(map(abs, pear_coeffs)))
            best_idx = non_check_rows[best_idx]
            best_maches.append(best_idx)  # Assign to that row the maximum pearcoeff
            non_check_rows.remove(best_idx)

        # Return the result in the new order
        result = permuted_data[:, best_maches]

        if axis == 0:
            result.transpose()

        return result

    def __call__(self):
        self.varimax_plus()
        return self.results


class Varimax:
    """
    Functions to perform a regular varimax
    """

    def __init__(self, data, truncate_by='max_comps', max_comps=60, fraction_explained_variance=0.9, verbose=True):

        self.data = data
        self.truncate_by = truncate_by
        self.max_comps = max_comps
        self.fraction_explained_variance = fraction_explained_variance
        self.verbose = verbose
        self.result = None

    def __call__(self):

        self.result = self.get_varimax_loadings_standard(self.data, self.truncate_by, self.max_comps,
                                                         self.fraction_explained_variance, self.verbose)
        return self.result

    def _pca_svd(self, data, truncate_by='max_comps', max_comps=60, fraction_explained_variance=0.9, verbosity=0):
        """

        :param data:
        :param truncate_by:
        :param max_comps:
        :param fraction_explained_variance:
        :param verbosity:
        :return:
        """

        """Assumes data of shape (obs, vars).

        https://stats.stackexchange.com/questions/134282/relationship-between-svd-
        and-pca-how-to-use-svd-to-perform-pca

        SVD factorizes the matrix A into two unitary matrices U and Vh, and a 1-D
        array s of singular values (real, non-negative) such that A == U*S*Vh,
        where S is a suitably shaped matrix of zeros with main diagonal s.

        K = min (obs, vars)

        U are of shape (vars, K)
        Vh are loadings of shape (K, obs)

        """

        # The actual function starts ####

        n_obs = data.shape[0]

        # Center data
        data -= data.mean(axis=0)

        # data_T = np.fastCopyAndTranspose(data)
        # print data.shape

        U, s, Vt = np.linalg.svd(data,
                                 full_matrices=False)
        # False, True, True)

        # flip signs so that max(abs()) of each col is positive
        U, Vt = self._svd_flip(U, Vt, u_based_decision=False)

        V = Vt.T
        S = np.diag(s)

        # eigenvalues of covariance matrix
        eig = (s ** 2) / (n_obs - 1.)

        # Sort
        idx = eig.argsort()[::-1]
        eig, U = eig[idx], U[:, idx]

        if truncate_by == 'max_comps':

            U = U[:, :max_comps]
            V = V[:, :max_comps]
            S = S[0:max_comps, 0:max_comps]
            explained = np.sum(eig[:max_comps]) / np.sum(eig)

        elif truncate_by == 'fraction_explained_variance':
            # print np.cumsum(s2)[:80] / np.sum(s2)
            max_comps = np.argmax(np.cumsum(eig) / np.sum(eig) > fraction_explained_variance) + 1
            explained = np.sum(eig[:max_comps]) / np.sum(eig)

            U = U[:, :max_comps]
            V = V[:, :max_comps]
            S = S[0:max_comps, 0:max_comps]

        else:
            max_comps = U.shape[1]
            explained = np.sum(eig[:max_comps]) / np.sum(eig)

        # Time series
        ts = U.dot(S)

        return V, U, S, ts, eig, explained, max_comps

    def _svd_flip(self, u, v=None, u_based_decision=True):
        """Sign correction to ensure deterministic output from SVD.
        Adjusts the columns of u and the rows of v such that the loadings in the
        columns in u that are largest in absolute value are always positive.
        Parameters
        ----------
        u, v : ndarray
            u and v are the output of `linalg.svd` or
            `sklearn.utils.extmath.randomized_svd`, with matching inner dimensions
            so one can compute `np.dot(u * s, v)`.
        u_based_decision : boolean, (default=True)
            If True, use the columns of u as the basis for sign flipping.
            Otherwise, use the rows of v. The choice of which variable to base the
            decision on is generally algorithm dependent.
        Returns
        -------
        u_adjusted, v_adjusted : arrays with the same dimensions as the input.
        """
        if v is None:
            # rows of v, columns of u
            max_abs_rows = np.argmax(np.abs(u), axis=0)
            signs = np.sign(u[max_abs_rows, range(u.shape[1])])
            u *= signs
            return u

        if u_based_decision:
            # columns of u, rows of v
            max_abs_cols = np.argmax(np.abs(u), axis=0)
            signs = np.sign(u[max_abs_cols, range(u.shape[1])])
            u *= signs
            v *= signs[:, np.newaxis]
        else:
            # rows of v, columns of u
            max_abs_rows = np.argmax(np.abs(v), axis=1)
            signs = np.sign(v[range(v.shape[0]), max_abs_rows])
            u *= signs
            v *= signs[:, np.newaxis]

        return u, v

    def varimax(self, Phi, gamma=1.0, q=500,
                rtol=np.finfo(np.float32).eps ** 0.5,
                verbosity=0):
        """

        :param Phi: The V of the SVD
        :param gamma: if = 1, equals to Varimax. if = 0 c. if = k/2 equamax. if = p*(k-1)/(p+k-2) parsimax
        :param q: number of iterations, breaks if objective archived before
        :param rtol: parameter of the machine
        :param verbosity: verbose
        :return: Rotated Phi and Rotation matrix
        """

        p, k = Phi.shape
        R = np.eye(k)
        d = 0

        # print Phi
        for i in range(q):
            if verbosity > 1:
                if i % 10 == 0.:
                    print("\t\tVarimax iteration %d" % i)
            d_old = d
            Lambda = np.dot(Phi, R)
            u, s, vh = np.linalg.svd(np.dot(Phi.T, np.asarray(Lambda) ** 3
                                            - (gamma / float(p)) * np.dot(Lambda,
                                                                          np.diag(np.diag(np.dot(Lambda.T, Lambda))))))
            R = np.dot(u, vh)
            d = np.sum(s)
            if d_old != 0 and abs(d - d_old) / d < rtol:
                break

        return np.dot(Phi, R), R

    def get_varimax_loadings_standard(self, data,
                                      truncate_by='max_comps',
                                      max_comps=60,
                                      fraction_explained_variance=0.9,
                                      verbosity=True,
                                      ):
        if verbosity:
            print("Get Varimax components")
            print("\t Get SVD")

        data = np.reshape(data, (data.shape[0], np.prod(data.shape[1:])))  # flattening field of daily data

        # Get truncated SVD
        V, U, S, ts_svd, eig, explained, max_comps = self._pca_svd(data=data,
                                                                   truncate_by=truncate_by, max_comps=max_comps,
                                                                   fraction_explained_variance=fraction_explained_variance,
                                                                   verbosity=verbosity)
        # if verbose > 0:
        #     print("Explained variance at max_comps = %d: %.5f" % (max_comps, explained))

        if verbosity:
            if truncate_by == 'max_comps':

                print("\t User-selected number of components: %d\n"
                      "\t Explaining %.2f of variance" % (max_comps, explained))

            elif truncate_by == 'fraction_explained_variance':

                print("\t User-selected explained variance: %.2f of total variance\n"
                      "\t Resulting in %d components" % (explained, max_comps))

        if verbosity:
            print("\t Varimax rotation")
        # Rotate
        Vr, Rot = self.varimax(V, verbosity=verbosity)
        # Vr = V
        # Rot = np.diag(np.ones(V.shape[1]))
        # print Vr.shape
        Vr = self._svd_flip(Vr)

        if verbosity:
            print("\t Further metrics")
        # Get explained variance of rotated components
        s2 = np.diag(S) ** 2 / (data.shape[0] - 1.)

        # matrix with diagonal containing variances of rotated components
        S2r = np.dot(np.dot(np.transpose(Rot), np.matrix(np.diag(s2))), Rot)
        expvar = np.diag(S2r)

        sorted_expvar = np.sort(expvar)[::-1]
        # s_orig = ((Vt.shape[1] - 1) * s2) ** 0.5

        # reorder all elements according to explained variance (descending)
        nord = np.argsort(expvar)[::-1]
        Vr = Vr[:, nord]

        # Get time series of UNMASKED data
        comps_ts = data.dot(Vr)

        total_var = np.sum(np.var(data, axis=0))

        return {'weights': np.copy(Vr),
                'explained_var': sorted_expvar,
                'unrotated_weights': V,
                'explained': explained,
                'pca_eigs': eig,
                'comps_ts': comps_ts,
                'truncate_by': truncate_by,
                'max_comps': max_comps,
                'fraction_explained_variance': fraction_explained_variance,
                'total_var': total_var,
                }
