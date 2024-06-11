import math
import numpy as np
import cv2
from typing import List, Tuple, Union
from typing import List, NamedTuple, Union


CACHE_DIR = './base_ckpt'



PARTS = {
        'face': [1,2,3,4, 16, 47,48, ],
        'face_corrected': [1,2,3,4, 16, 47,48, ],
        'skin': [7,8 ],
        'hand': [7,8 ],
        'left_hand': [7], 
        'right_hand': [8], 
        'shoes': [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 32,33,], 
        'scene': [0],
        'left_shoes': [22, 24, 26, 28,  30], 
        'right_shoes': [23, 25, 27, 29, 31], 
    'top': [5,6,7,8,9,10,11,12,13,14,15,16,34,36,37,38,40,57],
    'bottom': [17,18,19,30,31, 35,36, 37,38, 58],
            'face-hair': [1,2,3, 16, 47,48, ],


    }




SEG_IND_NAME_MAPPING = {
    0: 'Background',
    1: 'Cap/Hat',
    2: 'Helmet',
    3: 'Face',
    4: 'Hair',
    5: 'Left-arm',
    6: 'Right-arm',
    7: 'Left-hand',
    8: 'Right-hand',
    9: 'Protector',
    10: 'Bikini/Bra',
    11: 'Jacket/Windbreaker/Hoodie',
    12: 'T-shirt',
    13: 'Polo-shirt',
    14: 'Sweater',
    15: 'Singlet',
    16: 'Torso-skin',
    17: 'Pants',
    18: 'Shorts/Swim-shorts',
    19: 'Skirt',
    20: 'Stockings',
    21: 'Socks',
    22: 'Left-boot',
    23: 'Right-boot',
    24: 'Left-shoe',
    25: 'Right-shoe',
    26: 'Left-highheel',
    27: 'Right-highheel',
    28: 'Left-sandal',
    29: 'Right-sandal',
    30: 'Left-leg',
    31: 'Right-leg',
    32: 'Left-foot',
    33: 'Right-foot',
    34: 'Coat',
    35: 'Dress',
    36: 'Robe',
    37: 'Jumpsuits',
    38: 'Other-full-body-clothes',
    39: 'Headware',
    40: 'Backpack',
    41: 'Ball',
    42: 'Bats',
    43: 'Belt',
    44: 'Bottle',
    45: 'Carrybag',
    46: 'Cases',
    47: 'Sunglasses',
    48: 'Eyeware',
    49: 'Gloves',
    50: 'Scarf',
    51: 'Umbrella',
    52: 'Wallet/Purse',
    53: 'Watch',
    54: 'Wristband',
    55: 'Tie',
    56: 'Other-accessories',
    57: 'Other-Upper-Body-Clothes',
    58: 'Other-Lower-Body-Clothes'
}




class Keypoint(NamedTuple):
    x: float
    y: float
    score: float = 1.0
    id: int = -1


def get_folder_name(name, model_name, mask_type, mask_expand_ratio, guidance_scale):
    if model_name is None:
        return f'{name}/{mask_type}/{mask_expand_ratio}/{guidance_scale}'
        
    return f'{name}/{model_name}/{mask_type}/{mask_expand_ratio}/{guidance_scale}'


def draw_bodypose(canvas, keypoints) -> np.ndarray:
    """
    Draw keypoints and limbs representing body pose on a given canvas.

    Args:
        canvas (np.ndarray): A 3D numpy array representing the canvas (image) on which to draw the body pose.
        keypoints (List[Keypoint]): A list of Keypoint objects representing the body keypoints to be drawn.

    Returns:
        np.ndarray: A 3D numpy array representing the modified canvas with the drawn body pose.

    Note:
        The function expects the x and y coordinates of the keypoints to be normalized between 0 and 1.
    """
    H, W, C = canvas.shape
    stickwidth = 4

    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], 
        [6, 7], [7, 8], [2, 9], [9, 10], 
        [10, 11], [2, 12], [12, 13], [13, 14], 
        [2, 1], [1, 15], [15, 17], [1, 16], 
        [16, 18],
    ]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for (k1_index, k2_index), color in zip(limbSeq, colors):
        keypoint1 = keypoints[k1_index - 1]
        keypoint2 = keypoints[k2_index - 1]

        if keypoint1 is None or keypoint2 is None:
            continue

        Y = np.array([keypoint1.x, keypoint2.x]) * float(W)
        X = np.array([keypoint1.y, keypoint2.y]) * float(H)
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])

    for keypoint, color in zip(keypoints, colors):
        if keypoint is None:
            continue

        x, y = keypoint.x, keypoint.y
        x = int(x * W)
        y = int(y * H)
        cv2.circle(canvas, (int(x), int(y)), 4, color, thickness=-1)

    return canvas