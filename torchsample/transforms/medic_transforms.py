import numpy as np
import torch as th

'''
Transforms specific to biomedical images
'''


class NormalizeMedicPercentile(object):
    """
    Given min_val: float and max_val: float,
    will normalize each channel of the th.*Tensor to
    the provided min and max values.

    Works by calculating :
        a = (max'-min')/(max-min)
        b = max' - a * max
        new_value = a * value + b
    where min' & max' are given values,
    and min & max are observed min/max for each channel
    """

    def __init__(self,
                 min_val=0.0,
                 max_val=1.0,
                 perc_threshold=(1.0, 95.0),
                 norm_flag=True):
        """
        Normalize a tensor between a min and max value
        :param min_val: (float) lower bound of normalized tensor
        :param max_val: (float) upper bound of normalized tensor
        :param perc_threshold: (float, float) percentile of image intensities used for scaling
        :param norm_flag: [bool] list of flags for normalisation
        """

        self.min_val = min_val
        self.max_val = max_val
        self.perc_threshold = perc_threshold
        self.norm_flag = norm_flag

    def __call__(self, *inputs):
        # prepare the normalisation flag
        if isinstance(self.norm_flag, bool):
            norm_flag = [self.norm_flag] * len(inputs)
        else:
            norm_flag = self.norm_flag

        outputs = []
        for idx, _input in enumerate(inputs):
            if norm_flag[idx]:
                # determine the percentiles and threshold the outliers
                _min_val, _max_val = np.percentile(_input.numpy(), self.perc_threshold)
                _input[th.le(_input, _min_val)] = _min_val
                _input[th.ge(_input, _max_val)] = _max_val
                # scale the intensity values
                a = (self.max_val - self.min_val) / (_max_val - _min_val)
                b = self.max_val - a * _max_val
                _input = _input.mul(a).add(b)
            outputs.append(_input)

        return outputs if idx >= 1 else outputs[0]


class NormalizeMedic(object):
    """
    Normalises given slice/volume to zero mean
    and unit standard deviation.
    """

    def __init__(self,
                 norm_flag=True):
        """
        :param norm_flag: [bool] list of flags for normalisation
        """
        self.norm_flag = norm_flag

    def __call__(self, *inputs):
        # prepare the normalisation flag
        if isinstance(self.norm_flag, bool):
            norm_flag = [self.norm_flag] * len(inputs)
        else:
            norm_flag = self.norm_flag

        outputs = []
        for idx, _input in enumerate(inputs):
            if norm_flag[idx]:
                # subtract the mean intensity value
                mean_val = np.mean(_input.numpy().flatten())
                _input = _input.add(-1.0 * mean_val)

                # scale the intensity values to be unit norm
                std_val = np.std(_input.numpy().flatten())
                _input = _input.div(float(std_val))

            outputs.append(_input)

        return outputs if idx >= 1 else outputs[0]

