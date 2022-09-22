import torch
from torch.nn import functional as F

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel


@MODEL_REGISTRY.register()
class CATModle(SRModel):

    def test(self):
        self.use_chop = self.opt['val']['use_chop'] if 'use_chop' in self.opt['val'] else False
        if not self.use_chop:
            # pad to multiplication of window_size
            patch_size1 = max(self.opt['network_g']['split_size_0'])
            patch_size2 = max(self.opt['network_g']['split_size_1'])
            patch_size = max(patch_size1, patch_size2)
            
            scale = self.opt.get('scale', 1)
            mod_pad_h, mod_pad_w = 0, 0
            _, _, h, w = self.lq.size()
            if h % patch_size != 0:
                mod_pad_h = patch_size - h % patch_size
            if w % patch_size != 0:
                mod_pad_w = patch_size - w % patch_size
            img = self.lq
            img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h+mod_pad_h, :]
            img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w+mod_pad_w]
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                with torch.no_grad():
                    self.output = self.net_g_ema(img)
            else:
                self.net_g.eval()
                with torch.no_grad():
                    self.output = self.net_g(img)
                self.net_g.train()

            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

        else:
            _, _, h, w = self.lq.size()
            split_token_h = 1
            split_token_w = 1
            if h > 400:
                split_token_h = 2
            if w > 400:
                split_token_w = 2

            patch_size1 = max(self.opt['network_g']['split_size_0'])
            patch_size2 = max(self.opt['network_g']['split_size_1'])
            patch_size = max(patch_size1, patch_size2)

            patch_size_tmp_h = split_token_h * patch_size
            patch_size_tmp_w = split_token_w * patch_size

            mod_pad_h, mod_pad_w = 0, 0
            if h % patch_size_tmp_h != 0:
                mod_pad_h = patch_size_tmp_h - h % patch_size_tmp_h
            if w % patch_size_tmp_w != 0:
                mod_pad_w = patch_size_tmp_w - w % patch_size_tmp_w
                
            img = self.lq
            img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h+mod_pad_h, :]
            img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w+mod_pad_w]
            _, _, H, W = img.size()
            split_h = H // split_token_h
            split_w = W // split_token_w
            scale = self.opt.get('scale', 1)

            ral = H // split_h
            row = W // split_w
            slices = []
            for i in range(ral):
                for j in range(row):
                    top = slice(i*split_h, (i+1)*split_h)
                    left = slice(j*split_w, (j+1)*split_w)
                    temp = (top, left)
                    slices.append(temp)
            img_chops = []
            for temp in slices:
                top, left = temp
                img_chops.append(img[..., top, left])
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                with torch.no_grad():
                    outputs = []
                    for chop in img_chops:
                        out = self.net_g_ema(chop)
                        outputs.append(out)
                    _img = torch.zeros(1, 1, H*scale, W*scale)
                    for i in range(ral):
                        for j in range(row):
                            top = slice(i * split_h*scale, (i + 1) * split_h*scale)
                            left = slice(j * split_w*scale, (j + 1) * split_w*scale)
                            _img[..., top, left] = outputs[i*row+j]
                    self.output = _img
            else:
                self.net_g.eval()
                with torch.no_grad():
                    outputs = []
                    for chop in img_chops:
                        out = self.net_g(chop)
                        outputs.append(out)
                    _img = torch.zeros(1, 1, H*scale, W*scale)
                    for i in range(ral):
                        for j in range(row):
                            top = slice(i * split_h*scale, (i + 1) * split_h*scale)
                            left = slice(j * split_w*scale, (j + 1) * split_w*scale)
                            _img[..., top, left] = outputs[i*row+j]
                    self.output = _img
                self.net_g.train()
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

