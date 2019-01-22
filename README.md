# Face-preprocessing-tool
It's an useful tool that can automaticly do face detection, face crop and face rotate.

The base model used for face detection is [MTCNN](https://github.com/pangyupo/mxnet_mtcnn_face_detection). You can directly use it without training.

### Requirments

### Test
run:

`python main.py --root_path image/ --rotate_path images_rotate --crop_path images_crop`

The rotated image will be save in image_rotate.
The cropped image will be saved in image_crop.

### Results
<img src="https://github.com/wanfb/Face-Preprocessing-Tool/blob/master/image/Jay.jpg"/>
<img src="https://github.com/wanfb/Face-Preprocessing-Tool/blob/master/images_rotate/Jay.jpg"/>
<img src="https://github.com/wanfb/Face-Preprocessing-Tool/blob/master/images_crop/Jay.jpg"/>
