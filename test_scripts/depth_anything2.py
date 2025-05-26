import cv2
import numpy as np
import torch
import time
from act_plus_plus.detr.foundation_models.depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitb' # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])

model.load_state_dict(torch.load(f'/home/wzf/Project/rlbench_imitation_learning/foundation_ckpt/depth_anything_v2_{encoder}.pth', map_location='cpu', weights_only=True))
model = model.to(DEVICE).eval()

for i in range(10):
    raw_img = cv2.imread('/home/wzf/Pictures/near_robot.jpeg')
    # raw_img2 = cv2.imread('/home/wzf/Pictures/far_robot.jpeg')
    # raw_img = np.array([raw_img1, raw_img2])
    start_time = time.time()
    depth = model.infer_image(raw_img) # HxW raw depth map in numpy
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    end_time = time.time()
    print(f'inference time: {end_time - start_time:.2f}s')
    cv2.imshow("depth", depth_normalized)
    cv2.waitKey(0)