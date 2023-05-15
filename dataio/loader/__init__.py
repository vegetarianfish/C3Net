import json
from dataio.loader.cmr_3D_dataset import CMR3DDataset


def get_dataset(name):
    """get_dataset

    :param name:
    """
    return {
        'acdc_sax_edge': CMR3DDataset,
    }[name]


def get_dataset_path(dataset_name, opts):
    """get_data_path

    :param dataset_name:
    :param opts:
    """

    return getattr(opts, dataset_name)
