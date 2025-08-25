import cv2
import numpy as np

def overlay_mask_on_image(image, mask, color=(255, 0, 0), alpha=0.3):
    """
    将mask以指定颜色和透明度叠加到RGB图像上。
    
    参数:
    - image: RGB图像的NumPy数组,形状为(T, height, width, 3)
    - mask: 与图像形状相同的mask数组,形状为(T, height, width)
    - color: 叠加的颜色,默认为(255, 0, 0),表示红色
    - alpha: 叠加的透明度,默认为0.3
    
    返回:
    - overlaid_image: 叠加后的图像数组,形状与输入图像相同
    """
    T, height, width, _ = image.shape
    mask = np.array(mask)
    # 扩展mask的维度
    # mask = np.expand_dims(mask, axis=-1)
    # mask = np.tile(mask, (1, 1, 1, 3))
    if len(mask) == 0 or len(image) == 0:
        print(f'mask:{mask}, image:{image}', flush=True)
    # print(f'image.shape:{image[0].shape}, mask.shape:{mask[0].shape}, T:{T}', flush=True)
    overlaid_images = np.zeros_like(image, dtype=np.uint8)
    for t in range(T):
        # 创建一个与图像形状相同的颜色叠加层
        overlay = np.zeros((height, width, 3), dtype=np.uint8)
        overlay2 = np.zeros((height, width, 3), dtype=np.uint8)
        overlay[mask[0][t] == 1] = (255,0,0)  # 红色覆盖区域
        overlay2[mask[1][t] == 1] = (0,0,255)  # 蓝色覆盖区域
        
        # 将颜色叠加层和原始图像混合
        overlaid_image = cv2.addWeighted(overlay, alpha, image[t], 1 - alpha, 0)
        overlaid_image = cv2.addWeighted(overlay2, alpha, overlaid_image, 1 - alpha, 0)
        overlaid_images[t] = overlaid_image
    return overlaid_images

def overlay_masks(mask, T):
    """
    将mask重合在一起。
    
    参数:
    - mask: mask数组,形状为(2, height, width)
    - T: 与图像相同长度的mask
    
    返回:
    - overlaid_image: 叠加后的mask数组,形状与图像相同
    """
    mask = np.array(mask)

    # 扩展mask的维度
    if len(mask) == 0:
        print(f'mask:{mask} is none', flush=True)
    # print(f'mask[0].shape:{mask[0].shape}')
    _, height, width = mask[0].shape
    # print(f'image.shape:{image[0].shape}, mask.shape:{mask[0].shape}, T:{T}', flush=True)
    overlaid_images = np.zeros((T, height, width, 3), dtype=np.uint8)
    for t in range(T):
        # 创建一个与图像形状相同的颜色叠加层
        overlay = np.zeros((height, width, 3), dtype=np.uint8)
        overlay2 = np.zeros((height, width, 3), dtype=np.uint8)
        overlay[mask[0][t] == 1] = (255,0,0)  # 红色覆盖区域
        overlay2[mask[1][t] == 1] = (0,0,255)  # 蓝色覆盖区域
        
        overlaid_image = overlay + overlay2
        overlaid_images[t] = overlaid_image
    return overlaid_images

def overlay_box_on_image(image, box, color=(255, 0, 0)):
    """
    将边界框可视化在视频上，并保存结果视频。
    
    参数:
    - boxes: Numpy矩阵, 形状为 (frames, 8)，表示每帧的物体边界框，8代表两个边界框的左上角 (x1, y1) 和右下角 (x2, y2)
    """
    T, height, width, _ = image.shape
    images = []
    for t in range(T):
        x_min, y_min, x_max, y_max,  xx_min, yy_min, xx_max, yy_max = box[t] 
        # print('here:', x_min, y_min, x_max, y_max,  xx_min, yy_min, xx_max, yy_max)
        # 在帧上绘制边界框
        oimage = image[t]
        oimage = cv2.rectangle(oimage, (x_min, y_min), (x_max, y_max), (255,0,0), thickness=2)  
        oimage = cv2.rectangle(oimage, (xx_min, yy_min), (xx_max, yy_max), (0,0,255), thickness=2)  
        images.append(oimage)
    return images