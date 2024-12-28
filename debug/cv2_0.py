import os
import sys
sys.path.append('/home/chenyuheng/SelfBlendedImages/src')

from utils.sbi import SBI_Dataset

train_dataset=SBI_Dataset(phase='train',image_size=380)

train_dataset.__getitem__(173)