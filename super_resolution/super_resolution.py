import torch
import math
import numpy as np
from ai.settings.settings import BASE_DIR
import os

#
CFG = {
    'IMG_SIZE': 64,
    'BATCH_SIZE': 1,
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu'
}


class Super_Resolution():
    # SR with SwinIR model https://github.com/JingyunLiang/SwinIR

    def __init__(self):
        # load pretrained model
        self.model = torch.load(os.path.join(BASE_DIR, 'super_resolution', 'saved_models',
                                'model.pth'), map_location=CFG['DEVICE']).eval().to(CFG['DEVICE'])

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

    def inference(self, img_dict):
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
                img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [
                                      2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
                img_lq = torch.from_numpy(img_lq).float().unsqueeze(
                    0).to(CFG['DEVICE'])  # add .to() if u have cuda
                output = self.test(img_lq, window_size, SCALE)
                output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                if output.ndim == 3:
                    # CHW-RGB to HCW-BGR
                    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
                output = (output * 255.0).round().astype(np.uint8)
            return_dic[key] = output
        return return_dic
