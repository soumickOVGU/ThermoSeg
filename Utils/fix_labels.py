import os
from glob import glob

import nibabel as nib
import numpy as np
from skimage.filters import median
from skimage.morphology import disk
from tqdm import tqdm

# label_dir = '/project/schatter/Roopz/Segment/Data'
# label_files = glob(os.path.join(label_dir, '*_raw/**/*label.nii.gz'), recursive=True)

# for label_file in tqdm(label_files):
#     label = nib.load(label_file).get_fdata()
#     for i in range(label.shape[-1]):
#         label[..., i] = median(label[..., i], disk(4))
#     new_file_name = label_file.replace('_raw', '')
#     os.makedirs(os.path.dirname(new_file_name), exist_ok=True)
#     nib.save(nib.Nifti1Image(label, affine=np.eye(4)), new_file_name)

# label_dir = '/project/schatter/Roopz/Segment/Data'
# label_files = glob(os.path.join(label_dir, '*_label/**/*label.nii.gz'), recursive=True)

# for label_file in tqdm(label_files):
#     label = nib.load(label_file)
#     if label.shape != (128, 128, 25):
#         print(label_file)
    # if not np.array_equal(label.affine, np.eye(4)):
    #     print(np.unique(label))

#To fix the issue with subject 38's one session - size was wrong for Slicer output
p = "/project/schatter/Roopz/Segment/Data/train_label_raw/38/20170920/dynamic/EchoTime_19.1/Series06-Tumour-label.nii.gz"
label = nib.load(p).get_fdata()
from skimage.transform import resize
label_ = resize(label, (128,128,25), order=0, preserve_range=True)
for i in range(label_.shape[-1]):
    label_[..., i] = median(label_[..., i], disk(4))
p = "/project/schatter/Roopz/Segment/Data/train_label/38/20170920/dynamic/EchoTime_19.1/Series06-Tumour-label.nii.gz"
nib.save(nib.Nifti1Image(label_, affine=np.eye(4)), p)