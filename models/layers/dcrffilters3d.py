from abc import ABC, abstractmethod

import numpy as np
import torch

try:
    import permuto_cpp
except ImportError as e:
    raise (e, "Did you import `torch` first?")

_CPU = torch.device("cpu")
_EPS = np.finfo("float").eps


class PermutoFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q_in, features):
        q_out = permuto_cpp.forward(q_in, features)[0]
        ctx.save_for_backward(features)
        return q_out

    @staticmethod
    def backward(ctx, grad_q_out):
        feature_saved = ctx.saved_tensors[0]
        grad_q_back = permuto_cpp.backward(
            grad_q_out.contiguous(), feature_saved.contiguous()
        )[0]
        return grad_q_back, None  # No need of grads w.r.t. features


def _spatial_features(shape, sigma):
    """
    Return the spatial features as a Tensor
    Args:
        image:  Image as a Tensor of shape (channels, height, wight)
        sigma:  Bandwidth parameter
    Returns:
        Tensor of shape [h, w, 2] with spatial features
    """
    sigma = float(sigma)
    # C, D, W, H = image.shape
    # shape = (D, W, H)
    sdims = [sigma for _ in shape]

    # create mesh
    hcord_range = [range(s) for s in shape]
    mesh = np.array(np.meshgrid(*hcord_range, indexing='ij'), dtype=np.float32)

    # scale mesh accordingly
    for i, s in enumerate(sdims):
        mesh[i] /= s
    mesh = np.einsum('abcd->bcda', mesh)
    # mesh = np.transpose(mesh, (3,2,1,0))
    # return mesh.reshape([len(sdims), -1])

    # _, d, w, h = image.size()
    # x = torch.arange(start=0, end=w, dtype=torch.float32, device=_CPU)
    # xx = x.repeat([h, 1]) / sigma
    #
    # y = torch.arange(
    #     start=0, end=h, dtype=torch.float32, device=torch.device("cpu")
    # ).view(-1, 1)
    # yy = y.repeat([1, w]) / sigma

    # return torch.stack([xx, yy], dim=2)

    return torch.from_numpy(mesh).float()

class AbstractFilter(ABC):
    """
    Super-class for permutohedral-based Gaussian filters
    """

    def __init__(self, image):
        self.features = self._calc_features(image)
        self.norm = self._calc_norm(image).cuda()

    def apply(self, input_):
        input_ = input_.cpu()
        output = PermutoFunction.apply(input_, self.features)
        output = output.cuda()
        return output * self.norm

    @abstractmethod
    def _calc_features(self, image):
        pass

    def _calc_norm(self, shape):
        try:
            d, w, h = shape
        except:
            _, d, w, h = shape.size()
        all_ones = torch.ones((1, d, w, h), dtype=torch.float32, device=_CPU)
        norm = PermutoFunction.apply(all_ones, self.features)
        return 1.0 / (norm + _EPS)


class SpatialFilter(AbstractFilter):
    """
    Gaussian filter in the spatial ([x, y]) domain
    """

    def __init__(self, image, gamma):
        """
        Create new instance
        Args:
            image:  Image tensor of shape (3, height, width)
            gamma:  Standard deviation
        """
        self.gamma = gamma
        super(SpatialFilter, self).__init__(image)

    def _calc_features(self, image):
        return _spatial_features(image, self.gamma)


class BilateralFilter(AbstractFilter):
    """
    Gaussian filter in the bilateral ([r, g, b, x, y]) domain
    """

    def __init__(self, image, alpha, beta):
        """
        Create new instance
        Args:
            image:  Image tensor of shape (3, height, width)
            alpha:  Smoothness (spatial) sigma
            beta:   Appearance (color) sigma
        """
        self.alpha = alpha
        self.beta = beta
        super(BilateralFilter, self).__init__(image)

    def _calc_features(self, image):
        xy = _spatial_features(
            image.size()[1:], self.alpha
        )  # TODO Possible optimisation, was calculated in the spatial kernel
        rgb = (image / float(self.beta)).permute(1,2,3,0).cpu()  # Channel last order
        return torch.cat([xy, rgb], dim=3)