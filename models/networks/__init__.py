from .unet_CT_V_edge_3D import *


def get_network(name, n_classes, in_channels=3, aspp_channel=0, feature_scale=4, tensor_dim='2D',
                nonlocal_mode='embedded_gaussian', attention_dsample=(2,2,2), aggregation_mode='concat',
                res_connect=False, dropout=0, gpu_ids=(0,), edge_input=True, edge_type='binary'):
    model = get_model_instance(name, tensor_dim)

    assert name == 'unet_ct_v_edge'
    model = model(n_classes=n_classes,
                    is_batchnorm=True,
                    in_channels=in_channels, aspp_channel=aspp_channel,
                    nonlocal_mode=nonlocal_mode,
                    feature_scale=feature_scale,
                    attention_dsample=attention_dsample,
                    is_deconv=False, res_connect=res_connect,
                    dropout=dropout, gpu_ids=gpu_ids,
                    edge_input=edge_input, edge_type=edge_type)
    return model


def get_model_instance(name, tensor_dim):
    return {
        'unet_ct_v_edge': {'3D': unet_CT_V_edge_3D},
    }[name][tensor_dim]
