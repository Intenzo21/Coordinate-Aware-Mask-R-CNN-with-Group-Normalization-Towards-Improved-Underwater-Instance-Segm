"""
Script with additional auxiliary non-essential functions.
"""

import json
import os
from collections import defaultdict
from glob import glob
from tqdm import tqdm

"""
JSON file format:
    {
        'version': '4.5.13',
        'flags': {},
        'shapes': [{
            'label': 'dab',
            'points': [[1023.6559139784946, 865.5913978494623], [989.247311827957, 851.6129032258065], ...],
            'group_id': None,
            'shape_type': 'polygon',
            'flags': {}
        }],
        'imagePath': '05654_D20190719-T111209.744_18563406.jpg',
        'imageData': '...',
        'imageHeight': 1536,
        'imageWidth': 2048
    }
"""


def change_label(from_cls, to_cls, json_paths):
    """
    Change the instance label of the given JSON annotation files.

    :param from_cls: the label (class) to be changed
    :param to_cls: the target label (class)
    :param json_paths: JSON file paths
    :return: None
    """
    for j_path in tqdm(json_paths):
        with open(j_path, "r") as jf_read:
            annots = json.load(jf_read)

        img_shapes = annots["shapes"]

        for shp in img_shapes:
            cls = shp["label"]
            if cls in from_cls:
                shp["label"] = to_cls

        with open(j_path, "w") as jf_write:
            json.dump(annots, jf_write, indent=2)


def correct_pts(json_paths):
    """
    Adjust instance annotation points of the give JSON files.

    :param json_paths: JSON file paths
    :return: None
    """
    for j_path in tqdm(json_paths):
        with open(j_path, "r") as jf_read:
            annots = json.load(jf_read)

        img_shapes = annots["shapes"]

        for shp in img_shapes:
            pts = shp['points']
            for i, (x, y) in enumerate(pts):
                if 2042 < x < 2047:
                    x = 2047.0
                elif 0 < x < 5:
                    x = 0.0
                if 1530 < y < 1535:
                    y = 1535.0
                elif 0 < y < 5:
                    y = 0.0

                shp['points'][i] = [x, y]

        with open(j_path, "w") as jf_write:
            json.dump(annots, jf_write, indent=2)


def unpack(iterable):
    """
    Unpack a list by concatenating the elements with a comma.

    :param iterable: an iterable object (list, tuple, etc.)
    :return:
    """
    return ', '.join(str(x) for x in iterable)


def check_paths(json_paths, img_paths, check_label=None):
    """
    Check if JSON file names correspond to the image paths in the files.

    Also, check if instance shape points are in the image resolution
    range and count instances by class.

    :param json_paths: JSON file paths
    :param img_paths: image file paths
    :param check_label: label to check if present in the JSON files
    :return: image references and the instance counts by class
    """

    class_names = []
    img_heights = set()
    img_widths = set()
    img_refs = []

    class_counter = defaultdict(int)
    lbl_count = 0
    for j in json_paths:
        f = open(j)
        data = json.load(f)

        img_h, img_w = data['imageHeight'], data['imageWidth']
        img_heights.add(img_h)
        img_widths.add(img_w)
        img_ref = data['imagePath']
        #         print(j[:-5], data['imagePath'][:-4])
        # Check for reference duplicates
        j_fname = os.path.basename(j[:-5])

        if img_ref in img_refs:
            print(j_fname, data['imagePath'][:-4])

        img_refs.append(img_ref)

        # List of dictionaries corresponding to each shape
        img_shapes = data['shapes']

        # Add images
        for s in img_shapes:
            for x, y in s['points']:
                if isinstance(x, int) or isinstance(y, int):
                    print(img_ref)
                    print(f"Integer coordinates present: {x, y}")
                    return None, None
                if not (0 <= x <= 2047) or not (0 <= y <= 1535):
                    print(img_ref)
                    print(x, y)
                    return None, None

            label = s['label']

            if check_label and label in check_label:
                print(j)

            if label not in class_names:
                class_names.append(label)

            class_counter[label] += 1

    print(f"JSON file paths count: {len(json_paths)}")
    print(f"JSON image reference count: {len(img_refs)}")
    print(f"Image paths count: {len(img_paths)}")
    print(f"\nFish types (count = {len(class_names)}): {class_names}")
    print(f"\nImage heights: {unpack(img_heights)}")
    print(f"Image widths: {unpack(img_widths)}\n")

    return img_refs, class_counter


def compare_paths(json_paths, img_refs):
    """
    Compare JSON file names with their JPG image references which should be the same.

    :param json_paths: JSON file paths
    :param img_refs: image file references
    :return: None
    """

    assert len(json_paths) == len(img_refs)
    print(len(json_paths), len(img_refs))

    # ALL WORKS HERE
    for p1, p2 in zip(json_paths, img_refs):
        p1_fname, p2_fname = (os.path.basename(p1))[:-4], p2[:-4]
        if p1_fname != p2_fname:
            print(p1_fname, p2_fname)
            return
    print("ALL MATCH!")


def remove_img_data(json_paths):
    """
    Remove image data (not needed) to reduce JSON file size.

    :param json_paths: JSON file paths
    :return: None
    """
    for j_path in tqdm(json_paths):
        with open(j_path, "r") as jf_read:
            annots = json.load(jf_read)

        annots["imageData"] = None

        with open(j_path, "w") as jf_write:
            json.dump(annots, jf_write, indent=2)


# if __name__ == "__main__":
#     json_p = glob("data/test/json/*.json")
#     img_p = glob("data/test/raw/*.jpg")
#     remove_img_data(json_p)
#     change_label("goby", "flatfish", json_p)
#
#     img_refs, cls_counter = check_paths(json_paths=json_p, img_paths=img_p, check_label=["unidentified fish"])
#     for cls, count in cls_counter.items():
#         print(cls, count)
#     print(f"\nTotal {sum(cls_counter.values())} instances in {len(json_p)} images!")
#
#     compare_paths(img_p, img_refs)
#     correct_pts(json_p)

