import cv2 
import numpy as np

mask_path = 'mask_tensor_resize.jpg'


mask = cv2.imread(mask_path)

mask = mask < 200


# save 
mask = mask.astype(np.float32)
mask = mask * 255
mask = mask.astype(np.uint8)

cv2.imwrite('mask_tensor_resize_after.png', mask)

