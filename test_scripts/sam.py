import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
import time

# 加载模型
sam_checkpoint = "/home/madoka/python/sam_vit_h_4b8939.pth"  
model_type = "vit_b"

device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)

predictor = SamPredictor(sam)

# 读入一张图像
image = cv2.imread("0.png")
if image is None:
    print("错误：无法读取图像文件 '0.png'")
    exit()
    
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictor.set_image(image_rgb)

# 全局变量，用于存储点击位置和显示
clicked_point = None
window_name = 'SAM Interactive Segmentation'
display_image = image.copy()
result_image = np.zeros_like(image)  # 初始结果图像为空
combined_image = None  # 初始化组合图像

# 创建一个宽度足够的画布来放置两张并排的图像
def create_combined_image():
    h, w = image.shape[:2]
    combined = np.zeros((h, w*2, 3), dtype=np.uint8)
    combined[:, :w] = display_image
    combined[:, w:] = result_image
    # 添加分隔线
    combined[:, w-1:w+1] = [0, 0, 0]  # 黑色分隔线
    return combined

# 鼠标回调函数
def mouse_callback(event, x, y, flags, param):
    global clicked_point, display_image, combined_image
    
    # 只处理左半部分的点击
    h, w = image.shape[:2]
    if x >= w:  # 如果点击在右半部分，忽略
        return
        
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        # 重置显示图像
        display_image = image.copy()
        clicked_point = [x, y]
        # 在图像上标记点击位置
        cv2.circle(display_image, (x, y), 5, (0, 255, 0), -1)
        # 更新组合图像
        combined_image = create_combined_image()
        cv2.imshow(window_name, combined_image)
        print(f"已选择点: ({x}, {y})")

# 创建初始组合图像
combined_image = create_combined_image()

# 创建窗口并设置鼠标回调
cv2.namedWindow(window_name)
cv2.imshow(window_name, combined_image)
cv2.waitKey(100)  # 确保窗口已创建
cv2.setMouseCallback(window_name, mouse_callback)

print("请在左侧图像上点击要分割的物体，点击后按Enter键确认，按Q或ESC退出程序")

# 主循环
running = True
while running:
    # 等待用户点击并按下键盘
    clicked_point = None  # 重置点击状态
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13 and clicked_point is not None:  # Enter键且已有点击
            break
        elif key == ord('q') or key == 27:  # q键或ESC退出
            running = False
            break
    
    if not running:
        break
        
    # 将点击位置转换为numpy数组格式
    input_point = np.array([clicked_point])
    input_label = np.array([1])  # 1 表示前景

    # 测量推理时间
    start_time = time.time()
    # 获取 mask
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    inference_time = time.time() - start_time
    print(f"推理完成，耗时: {inference_time:.3f} 秒，置信度: {scores[0]:.3f}")
    
    # 创建分割结果图像
    mask = masks[0]
    # 将RGB图像转换回BGR用于OpenCV显示
    result_rgb = image_rgb.copy()
    
    # 给分割区域添加颜色叠加效果
    color_mask = np.zeros_like(result_rgb)
    color_mask[mask] = [0, 0, 255]  # 红色遮罩
    result_rgb = cv2.addWeighted(result_rgb, 1.0, color_mask, 0.5, 0)
    
    # 添加文字信息
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result_rgb, f"Confidence: {scores[0]:.3f}", (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(result_rgb, f"Time: {inference_time:.3f}s", (10, 60), font, 0.7, (255, 255, 255), 2)
    
    # 转换回BGR颜色格式
    result_image = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    
    # 更新组合图像并显示
    combined_image = create_combined_image()
    cv2.imshow(window_name, combined_image)
    cv2.waitKey(1)  # 刷新显示
    
    print("可继续在左侧图像上选择新的点进行分割，或按Q/ESC退出程序")

cv2.destroyAllWindows()
print("程序已退出")