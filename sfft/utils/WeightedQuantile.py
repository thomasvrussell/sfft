import warnings
import numpy as np
# version: Aug 17, 2022

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.3"

class Weighted_Quantile:
    @staticmethod
    def WQ(values, weights, quantiles, values_sorted=False, old_style=False):
        # ref: https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
        """ Very close to numpy.percentile, but supports weights.
        NOTE: quantiles should be in [0, 1]!
        :param values: numpy.array with data
        :param weights: array-like of the same length as `array`
        :param quantiles: array-like with many quantiles needed
        :param values_sorted: bool, if True, then will avoid sorting of
            initial array
        :param old_style: if True, will correct output to be consistent
            with numpy.percentile.
        :return: numpy.array with computed quantiles.
        """
        values = np.array(values)
        quantiles = np.array(quantiles)
        if weights is None:
            weights = np.ones(len(values))
        weights = np.array(weights)
        assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
            'quantiles should be in [0, 1]'

        if not values_sorted:
            sorter = np.argsort(values)
            values = values[sorter]
            weights = weights[sorter]

        weighted_quantiles = np.cumsum(weights) - 0.5 * weights
        if old_style:
            # To be convenient with numpy.percentile
            weighted_quantiles -= weighted_quantiles[0]
            weighted_quantiles /= weighted_quantiles[-1]
        else:
            weighted_quantiles /= np.sum(weights)
        return np.interp(quantiles, weighted_quantiles, values)

class TopFlatten_Weighted_Quantile:
    @staticmethod
    def TFWQ(values, weights, quantiles, NUM_TOP_END=30):
        # *** Note on TopFlatten Weighted ***
        # (1) We flatten the top weights to avoid the calculation strongly biased towards 
        #     to a handful of inidividual objects with extremely high weights.
        # (2) E.g., When you use source flux as sample weight, it is possible that a few 
        #     brightest sources in the field would possess undesired dramtically high weights.
        #     In such cases, it is useful to leverage the TopFlatten to calculate weighted quantiles.

        assert len(values) > 0
        if len(values) <= NUM_TOP_END:
            # flatten all when the sample size < NUM_TOP_END
            outputs = np.percentile(values, quantiles)
            _warn_message = 'CALCULATING WEIGHTED QUANTILES --- '
            _warn_message += 'USE UNIFORM-WEIGHTED MEDIAN OVER [%d] SAMPLES!' %(len(values))
            warnings.warn('MeLOn WARNING: %s' %_warn_message)
            return outputs
        
        # flatten the top end
        topflatten = lambda W: np.clip(W/(np.sort(W)[-NUM_TOP_END]), a_min=0.0, a_max=1.0)
        outputs = Weighted_Quantile.WQ(values=values, quantiles=quantiles, weights=topflatten(weights))
        return outputs
