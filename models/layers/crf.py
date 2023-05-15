import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils


class DenseCRF3D(object):
    def __init__(self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std):
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std

    def __call__(self, image, probmap):
        C, D, W, H = probmap.shape
        shape = (D, W, H)

        U = utils.unary_from_softmax(probmap)
        U = np.ascontiguousarray(U)

        image = np.ascontiguousarray(image)

        d = dcrf.DenseCRF(np.prod(shape), C)
        d.setUnaryEnergy(U)

        featsGaussian = utils.create_pairwise_gaussian(sdims=(self.pos_xy_std, self.pos_xy_std, self.pos_xy_std), shape=shape)
        d.addPairwiseEnergy(featsGaussian, compat=self.pos_w, kernel=dcrf.FULL_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

        featsBilateral = utils.create_pairwise_bilateral(sdims=(self.bi_xy_std, self.bi_xy_std, self.bi_xy_std), schan=self.bi_rgb_std, img=image, chdim=0)
        d.addPairwiseEnergy(featsBilateral, compat=self.bi_w, kernel=dcrf.FULL_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

        # d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        # d.addPairwiseBilateral(
        #     sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w
        # )

        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape((C, D, W, H))

        return Q