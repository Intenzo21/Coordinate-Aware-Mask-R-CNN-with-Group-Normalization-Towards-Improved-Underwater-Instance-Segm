from pathlib import Path
import cv2
import json

import retinex
from constants import TEST_IMG_DIR, IMG_DIR
from tqdm import tqdm

data_dir = Path(f"../../", "data/test/")
img_list = list(data_dir.glob("*jpg"))  # glob(f"{data_dir}*.jpg")

target_dir = Path("../../data/test/", "new_amsrcr")
target_dir.mkdir(exist_ok=True)

if len(img_list) == 0:
    print('Data directory is empty.')
    exit()

with open('config.json', 'r') as f:
    config = json.load(f)


for img_path in tqdm(img_list):
    # print(img_path)
    img = cv2.imread(str(img_path))

    # img_msrcr = retinex.MSRCR(
    #     img,
    #     config['sigma_list'],
    #     config['G'],
    #     config['b'],
    #     config['alpha'],
    #     config['beta'],
    #     config['low_clip'],
    #     config['high_clip']
    # )

    img_amsrcr = retinex.automatedMSRCR(
        img,
        config['sigma_list']
    )

    # img_msrcp = retinex.MSRCP(
    #     img,
    #     config['sigma_list'],
    #     config['low_clip'],
    #     config['high_clip']
    # )

    # shape = img.shape
    # cv2.imshow('Image', img)
    # cv2.imshow('retinex', img_msrcr)
    # cv2.imshow('Automated retinex', img_amsrcr)
    # cv2.imshow('MSRCP', img_msrcp)
    # cv2.imwrite("retinex.jpg", img_msrcr)

    target_path = Path(target_dir, img_path.name)
    cv2.imwrite(str(target_path), img_amsrcr)
    # cv2.imwrite("msrcp.jpg", img_msrcp)
    # cv2.waitKey(0)
