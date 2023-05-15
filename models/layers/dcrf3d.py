import torch
import torch.nn as nn

from models.layers.dcrffilters3d import SpatialFilter, BilateralFilter

class DenseCRFParams3d(object):
    """
    Parameters for the DenseCRF model
    """

    def __init__(
        self,
        alpha=67.0, #160.0,
        beta=3.0,
        gamma=1.0, #3.0,
        spatial_ker_weight=3.0,
        bilateral_ker_weight=5.0,
    ):
        """
        Default values were taken from https://github.com/sadeepj/crfasrnn_keras. More details about these parameters
        can be found in https://arxiv.org/pdf/1210.5644.pdf
        Args:
            alpha:                  Bandwidth for the spatial component of the bilateral filter
            beta:                   Bandwidth for the color component of the bilateral filter
            gamma:                  Bandwidth for the spatial filter
            spatial_ker_weight:     Spatial kernel weight
            bilateral_ker_weight:   Bilateral kernel weight
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.spatial_ker_weight = spatial_ker_weight
        self.bilateral_ker_weight = bilateral_ker_weight

class CrfRnn3d(nn.Module):
    """
    PyTorch implementation of the CRF-RNN module described in the paper:
    Conditional Random Fields as Recurrent Neural Networks,
    S. Zheng, S. Jayasumana, B. Romera-Paredes, V. Vineet, Z. Su, D. Du, C. Huang and P. Torr,
    ICCV 2015 (https://arxiv.org/abs/1502.03240).
    """

    def __init__(self, num_labels, num_iterations=5, crf_init_params=None):
        """
        Create a new instance of the CRF-RNN layer.
        Args:
            num_labels:         Number of semantic labels in the dataset
            num_iterations:     Number of mean-field iterations to perform
            crf_init_params:    CRF initialization parameters
        """
        super(CrfRnn3d, self).__init__()

        if crf_init_params is None:
            crf_init_params = DenseCRFParams3d()

        self.params = crf_init_params
        self.num_iterations = num_iterations

        self._softmax = torch.nn.Softmax(dim=0)

        self.num_labels = num_labels

        # --------------------------------- Trainable Parameters -------------------------------------

        # Spatial kernel weights
        self.spatial_ker_weights = nn.Parameter(
            crf_init_params.spatial_ker_weight
            * torch.eye(num_labels, dtype=torch.float32)
        )

        # Bilateral kernel weights
        self.bilateral_ker_weights = nn.Parameter(
            crf_init_params.bilateral_ker_weight
            * torch.eye(num_labels, dtype=torch.float32)
        )

        # Compatibility transform matrix
        self.compatibility_matrix = nn.Parameter(
            torch.eye(num_labels, dtype=torch.float32)
        )

    def forward(self, image, logits):
        """
        Perform CRF inference.
        Args:
            image:  Tensor of shape (3, h, w) containing the RGB image
            logits: Tensor of shape (num_classes, h, w) containing the unary logits
        Returns:
            log-Q distributions (logits) after CRF inference
        """
        if logits.shape[0] != 1:
            raise ValueError("Only batch size 1 is currently supported!")

        image = image[0]
        logits = logits[0]
        # print(image.size(), logits.size())
        # torch.Size([1, 144, 144, 144]), torch.Size([9, 144, 144, 144])

        spatial_filter = SpatialFilter(image.size()[1:], gamma=self.params.gamma)
        bilateral_filter = BilateralFilter(image, alpha=self.params.alpha, beta=self.params.beta)

        _, d, w, h = image.shape
        cur_logits = logits

        for _ in range(self.num_iterations):
            # Normalization
            q_values = self._softmax(cur_logits)

            # Spatial filtering
            spatial_out = torch.mm(
                self.spatial_ker_weights,
                spatial_filter.apply(q_values).view(self.num_labels, -1),
            )
            # print(self.spatial_ker_weights.size(), spatial_out.size())
            # torch.Size([9, 9]), torch.Size([9, 2985984])

            # Bilateral filtering
            bilateral_out = torch.mm(
                self.bilateral_ker_weights,
                bilateral_filter.apply(q_values).view(self.num_labels, -1),
            )

            # Compatibility transform
            msg_passing_out = (spatial_out + bilateral_out)  # Shape: (self.num_labels, -1)
            msg_passing_out = torch.mm(self.compatibility_matrix, msg_passing_out).view(self.num_labels, d, w, h)

            # Adding unary potentials
            cur_logits = msg_passing_out + logits

        return torch.unsqueeze(cur_logits, 0)

if __name__ == '__main__':
    depth=3
    batch_size=1
    # encoder = One_Hot(depth=depth).forward
    img = torch.randn(batch_size, 1, 3, 3, 3).float().cuda()
    logits = torch.randn(batch_size, depth, 3, 3, 3).float().cuda()
    crfrnn = CrfRnn3d(num_labels=depth).cuda()
    out = crfrnn(img, logits)
    # sm = nn.Softmax(dim=1)
    # out = sm(out).max(1)[1].unsqueeze(1).float()
    # print(out.size())
    y = torch.randn(batch_size, depth, 3, 3, 3).random_().cuda() % depth  # 4 classes,1x3x3 img
    loss = nn.MSELoss().cuda()
    err = loss(out, y)
    err.backward()

