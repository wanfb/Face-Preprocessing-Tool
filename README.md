# Face-preprocessing-tool
It's an useful tool that can automaticly do face detection, face crop and face rotate.

The base model used for face detection is [MTCNN](https://github.com/pangyupo/mxnet_mtcnn_face_detection). You can directly use it without training.

### Requirments
* asn1crypto==0.24.0
* certifi==2018.10.15
* cffi==1.11.5
* chardet==3.0.4
* cryptography==2.4.1
* enum34==1.1.6
* graphviz==0.8.4
* idna==2.7
* ipaddress==1.0.22
* mxnet==1.3.1
* numpy==1.14.6
* opencv-python==4.0.0.21
* PIL==1.1.7
* pycparser==2.19
* pyOpenSSL==18.0.0
* PySocks==1.6.8
* requests==2.21.0
* six==1.11.0
* urllib3==1.24.1

### Test
run:
`python main.py --root_path image/ --rotate_path images_rotate --crop_path images_crop`

The rotated image will be save in image_rotate.
The cropped image will be saved in image_crop.

### Results
<img src="https://github.com/wanfb/Face-Preprocessing-Tool/blob/master/image/Jay.jpg" width = "250" height = "250" />
<img src="https://github.com/wanfb/Face-Preprocessing-Tool/blob/master/images_rotate/Jay.jpg" width = "250" height = "250"/>
<img src="https://github.com/wanfb/Face-Preprocessing-Tool/blob/master/images_crop/Jay.jpg"/>
