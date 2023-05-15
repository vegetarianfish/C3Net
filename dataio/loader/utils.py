import nibabel as nib
import numpy as np
import os
from utils.util import mkdir

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".nii.gz"])


def load_nifti_img(filepath, dtype, reorient=True):
    '''
    NIFTI Image Loader
    :param filepath: path to the input NIFTI image
    :param dtype: dataio type of the nifti numpy array
    :return: return numpy array
    '''
    # if 'nifty' in filepath:
    #     img_no = int(filepath.split('__')[0][-3:])
    # elif 'ALL_' in filepath:
    #     img_no = int(filepath.split('ALL_')[-1][:3])
    # else:
    #     img_no = int(filepath.split('ct')[-1][:3])
    nim = nib.load(filepath)
    out_nii_array = np.array(nim.get_data(),dtype=dtype)
    out_nii_array = np.squeeze(out_nii_array) # drop singleton dim in case temporal dim exists
    # if reorient:
    #     if 1 <= img_no <= 43:
    #         pass
    #     elif 44 <= img_no <= 90:
    #         out_nii_array = np.flip(out_nii_array, 2)
    #     elif 91 <= img_no <= 110:
    #         out_nii_array = np.flip(out_nii_array, (0,1,2))
    #     else:
    #         raise (Exception('out of bound image index'))

    meta = {'affine': nim.get_affine(),
            'dim': nim.header['dim'],
            'pixdim': nim.header['pixdim'],
            'name': os.path.basename(filepath)
            }

    return out_nii_array, meta


def write_nifti_img(input_nii_array, meta, savedir):
    mkdir(savedir)
    affine = meta['affine'][0].cpu().numpy()
    pixdim = meta['pixdim'][0].cpu().numpy()
    dim    = meta['dim'][0].cpu().numpy()

    img = nib.Nifti1Image(input_nii_array, affine=affine)
    img.header['dim'] = dim
    img.header['pixdim'] = pixdim

    savename = os.path.join(savedir, meta['name'][0])
    print('saving: ', savename)
    nib.save(img, savename)


def check_exceptions(image, label=None):
    if label is not None:
        if image.shape != label.shape:
            print('Error: mismatched size, image.shape = {0}, '
                  'label.shape = {1}'.format(image.shape, label.shape))
            #print('Skip {0}, {1}'.format(image_name, label_name))
            raise(Exception('image and label sizes do not match'))

    if image.max() < 1e-6:
        print('Error: blank image, image.max = {0}'.format(image.max()))
        #print('Skip {0} {1}'.format(image_name, label_name))
        raise (Exception('blank image exception'))