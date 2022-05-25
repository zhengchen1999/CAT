from torch.utils import data as data
from torchvision.transforms.functional import normalize
import torchvision.transforms.functional as TF
import numpy as np
import cv2

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.matlab_functions import rgb2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class PairedImageDataset_Master(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
            n_max (int): only first n_max images
    """

    def __init__(self, opt):
        super(PairedImageDataset_Master, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.task = opt['task'] if 'task' in opt else 'sr'
        self.noise = opt['noise'] if 'noise' in opt else 0
        self.jpeg = opt['jpeg'] if 'jpeg' in opt else 100

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']


        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            if not isinstance(self.gt_folder, list):
                self.gt_folder, self.lq_folder = [self.gt_folder], [self.lq_folder]
            self.filename_tmpl = opt['filename_tmpl'] if 'filename_tmpl' in opt else ['{}']*len(self.lq_folder)
            self.filename_replace = opt['filename_replace'] if 'filename_replace' in opt else [['','']]*len(self.lq_folder)
            if not isinstance(self.filename_tmpl, list):
                self.filename_tmpl = [self.filename_tmpl]*len(self.lq_folder)
            if not isinstance(self.filename_replace[0], list):
                self.filename_replace = [self.filename_replace]*len(self.lq_folder)
            assert len(self.gt_folder) == len(self.lq_folder) == len(self.filename_tmpl) == len(self.filename_replace)

            self.paths = []
            for gt_folder, lq_folder, filename_tmpl, filename_replace in zip(self.gt_folder, self.lq_folder, self.filename_tmpl, self.filename_replace):
                self.paths += paired_paths_from_folder([lq_folder, gt_folder], ['lq', 'gt'], filename_tmpl)

        self.n_max = opt['n_max'] if 'n_max' in opt else int(1e15)
        self.paths = self.paths[: self.n_max]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        if self.task == 'sr':
            gt_path = self.paths[index]['gt_path']
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            lq_path = self.paths[index]['lq_path']
            img_bytes = self.file_client.get(lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)

        elif self.task == 'deblocking_gray':
            gt_path = self.paths[index]['gt_path']
            lq_path = gt_path
            img_bytes = self.file_client.get(gt_path, 'gt')
            # # OpenCV version
            # img_gt = imfrombytes(img_bytes, flag='grayscale', float32=False)
            # Matlab version, following "Compression Artifacts Reduction by a Deep Convolutional Network"
            img_gt = imfrombytes(img_bytes, flag='unchanged', float32=False)
            if img_gt.ndim != 2:
                img_gt = rgb2ycbcr(cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB), y_only=True)
            result, encimg = cv2.imencode('.jpg', img_gt, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg])
            img_lq = cv2.imdecode(encimg, 0)
            img_gt = np.expand_dims(img_gt, axis=2).astype(np.float32)/ 255.
            img_lq = np.expand_dims(img_lq, axis=2).astype(np.float32)/ 255.

        # crop HR images, added by jinliang
        img_gt = img_gt[:img_lq.shape[0]*scale, :img_lq.shape[1]*scale, :]

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'], self.opt['use_rot'])

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
