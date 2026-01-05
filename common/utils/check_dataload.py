import cv2
import numpy as np

# Added by SH Heo(260105)
def show_input_image(inputs, window_name='Input Image', wait_key=1):
    img_tensor = inputs['img']

    # [C, H, W]
    img = img_tensor[0].cpu().numpy()

    # [C, H, W] -> [H, W, C]
    img = img.transpose(1, 2, 0)

    # Restore normalization (if 0~1 range, convert to 0~255)
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    # RGB -> BGR for O
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imshow(window_name, img)
    cv2.waitKey(wait_key)