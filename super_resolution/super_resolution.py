import torch
import math
import numpy as np
from ai.settings.settings import BASE_DIR
import os
from super_resolution.modules.net import SwinIR
import cv2

CFG = {
    'IMG_SIZE': 64,
    'BATCH_SIZE': 1,
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'SCALE': 2
}


class Super_Resolution():
    # SR with SwinIR model https://github.com/JingyunLiang/SwinIR

    def __init__(self, light=True):
        # load pretrained model
        param_key_g = 'params'
        if light:
            model = SwinIR(upscale=CFG['SCALE'], in_chans=1, img_size=CFG['IMG_SIZE'], window_size=8,
                            img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                        mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
            pretrained_dict = torch.load(os.path.join(
                BASE_DIR, 'super_resolution', 'saved_models', 'SwinIR_light_gray_2x.pth'), map_location=torch.device(CFG['DEVICE']))
        else: 
            model = SwinIR(upscale=CFG['SCALE'], in_chans=3, img_size=CFG['IMG_SIZE'], window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
            pretrained_dict = torch.load(os.path.join(
                BASE_DIR, 'super_resolution', 'saved_models', 'SwinIR_2x.pth'))

        model.load_state_dict(pretrained_dict[param_key_g] if param_key_g in pretrained_dict.keys(
        ) else pretrained_dict, strict=True)
        self.model = model

    def test(self, img_lq, window_size, sf):
        b, c, h, w = img_lq.size()
        tile = min(h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"

        stride = tile
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = self.model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx *
                  sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx *
                  sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

        return output

    def inference(self, img_dict, light=True):
        window_size = 8
        SCALE = 2
        return_dic = {}
        for key in img_dict:
            img_lq = img_dict[key].astype(np.float32) / 255.
            with torch.no_grad():
                h_old, w_old, _ = img_lq.shape
                h_pad = math.ceil(h_old / window_size) * window_size - h_old
                w_pad = math.ceil(w_old / window_size) * window_size - w_old
                img_lq = np.pad(img_lq, ((0, h_pad), (0, w_pad), (0, 0)),
                                mode='constant', constant_values=np.median(img_lq))
                if light:
                    img_lq = cv2.cvtColor(img_lq, cv2.COLOR_RGB2GRAY)
                    img_lq = torch.from_numpy(img_lq).float().unsqueeze(
                        0).unsqueeze(0).to(CFG['DEVICE'])  # add .to() if u have cuda
                else:
                    img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [
                                        2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
                    img_lq = torch.from_numpy(img_lq).float().unsqueeze(
                        0).to(CFG['DEVICE'])  # add .to() if u have cuda
                output = self.test(img_lq, window_size, SCALE)
                output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                if light:
                    output = cv2.cvtColor((output * 255.0).round().astype(np.uint8), cv2.COLOR_GRAY2BGR)
                else:
                    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
                    output = (output * 255.0).round().astype(np.uint8)
            return_dic[key] = output
        return return_dic
