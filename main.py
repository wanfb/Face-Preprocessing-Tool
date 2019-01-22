# coding: utf-8
import argparse
from rotate_face import CropFace
import mxnet as mx
from mtcnn_detector import MtcnnDetector
import cv2
import os
import time
import pdb
import PIL.Image
import pdb

parser = argparse.ArgumentParser(description='Face Preprocessing Tool')
parser.add_argument('--root_path', default='', type=str, metavar='PATH',
                    help='path to path of raw images (default: none)')
parser.add_argument('--crop_path', default='', type=str, metavar='PATH',
                    help='path to path of cropped images (default: none)')
parser.add_argument('--rotate_path', default='', type=str, metavar='PATH',
                    help='path to path of rotate images (default: none)')
parser.add_argument('--num_workers', default=4, type=int,
                    metavar='N', help='number of workers (default: 4)')
parser.add_argument('--crop_size', default=0.37, type=int,
                    metavar='N', help='size of cropped image (default: 0.37)')


def main():
    global args
    args = parser.parse_args()
    detector = MtcnnDetector(model_folder='model', ctx=mx.cpu(0), num_worker = args.num_workers, accurate_landmark = False)

    for root, dirs, files in os.walk(args.root_path):
        try:
            if files[0].split('.')[1] != 'jpg':
                continue
        except IndexError:
            continue

        # make dirs
        if not os.path.exists(args.rotate_path):
            os.makedirs(args.rotate_path)
        if not os.path.exists(args.crop_path):
            os.makedirs(args.crop_path)

        for image in files:
            img = cv2.imread(root + '/' + image)
            
            # run detector first round
            results = detector.detect_face(img)

            if results is not None:
                total_boxes = results[0]
                points = results[1]
    
                leftx = points[0][0]
                lefty = points[0][5]
                rightx = points[0][1]
                righty = points[0][6]
 
                # rotate face
                img_tmp = PIL.Image.open(root + '/' + image)
                CropFace(img_tmp, eye_left=(leftx,lefty), eye_right=(rightx,righty), offset_pct=(0.1,0.1), dest_sz=(200,200)).save(args.rotate_path + '/' + image)

            # run detector second round
            draw = cv2.imread(args.rotate_path + '/' + image)
            results = detector.detect_face(draw)

            if results is not None:
                total_boxes = results[0]
                points = results[1]

                # extract aligned face chips
                chips = detector.extract_image_chips(draw, points, 144, args.crop_size)
                for i, chip in enumerate(chips):
                    cv2.imwrite(args.crop_path + '/' + image, chip)
                    print(args.crop_path + '/' + image)

if __name__ == '__main__':
    main()

