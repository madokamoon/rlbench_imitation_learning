from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import imageio
import matplotlib.pyplot as plt
import cv2
import numpy as np

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

device = "cuda"
image_path = "/home/starsky/project/gitee/rlbench_imitation_learning/data/pick_and_lift/20demos_static/0/overhead_camera/0.png"
image = imageio.imread(image_path)


sam = sam_model_registry["vit_h"](checkpoint="/home/starsky/project/gitee/rlbench_imitation_learning/foundation_ckpt/sam_vit_h_4b8939.pth")
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=64,
    pred_iou_thresh=0.90,
    stability_score_thresh=0.95,
    min_mask_region_area=100000,)
masks = mask_generator.generate(image)

print(f"mask len: {len(masks)}")
plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show()






