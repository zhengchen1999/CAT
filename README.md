# SR

conda install -c conda-forge 

numpy typing-extensions pillow

future lmdb Pillow pyyaml requests scikit-image scipy tqdm yapf

absl-py google-auth-oauthlib werkzeug grpcio tensorboard-plugin-wit tensorboard-data-server markdown protobuf

torch torchvision torchaudio opencv-python addict tb-nightly

pip lmdb wheel --force-reinstall  pyasn1 pyasn1_modules fonttools

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/y50021751/miniconda3/lib/

https://polybox.ethz.ch/index.php/login
https://docs.google.com/document/d/1zNb_RoyZmw7l3OGpOLUyzXvSCT3TpzBjdvVq_COs5vw/edit

29dba854298aa0ffe5d83d13c5b5981df8d00afa

# calculate attention mask for SW-MSA
img_mask_v = torch.zeros((1, H, self.split_size, 1))  # 1 H W 1 竖直
img_mask_h = torch.zeros((1, self.split_size, W, 1))  # 1 H W 1 水平
slices = (
            slice(-self.split_size, -self.shift_size),
            slice(-self.shift_size, None))
cnt = 0
for s in slices:
    img_mask_v[:, :, s, :] = cnt
    img_mask_h[:, s, :, :] = cnt
    cnt += 1

img_mask_v = img_mask_v.view(1, H // H, H, self.split_size // self.split_size, self.split_size, 1)
img_mask_h = img_mask_h.view(1, self.split_size // self.split_size, self.split_size, W // W, W, 1)
img_mask_v = img_mask_v.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, self.split_size, 1) # nW, H, sw, 1
img_mask_h = img_mask_h.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.split_size, W, 1) # nW, sw, W, 1

mask_windows_v = img_mask_v.view(-1, H * self.split_size)
mask_windows_h = img_mask_h.view(-1, self.split_size * W)

attn_mask_v = mask_windows_v.unsqueeze(1) - mask_windows_v.unsqueeze(2)
attn_mask_v = attn_mask_v.masked_fill(attn_mask_v != 0, float(-100.0)).masked_fill(attn_mask_v == 0, float(0.0))

attn_mask_h = mask_windows_h.unsqueeze(1) - mask_windows_h.unsqueeze(2)
attn_mask_h = attn_mask_h.masked_fill(attn_mask_h != 0, float(-100.0)).masked_fill(attn_mask_h == 0, float(0.0))

num_v = W // self.split_size
num_h = H // self.split_size

attn_mask_v_la = torch.zeros((num_v,H * self.split_size,H * self.split_size))  # 1 H W 1 竖直
attn_mask_h_la = torch.zeros((num_h,W * self.split_size,W * self.split_size))  # 1 H W 1 水平

attn_mask_v_la[-1] = attn_mask_v
attn_mask_h_la[-1] = attn_mask_h
return attn_mask_v_la, attn_mask_h_la
