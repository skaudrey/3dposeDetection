# 3D Pose Detection

## Task

This project is a demo for predicting 3D pose from monocular image captured by PC's camera in real time. There is an interaction between server and client. The basic ideas comes from hourglass and siamese 3d pose (Reference 1 and 2). Specifically, after connecting with client, once client send "capture" request, server will capture one image by PC's camera, then detect bounding boxes with pretrained YOLOV3. According to bounding box, server will compute each bounding box's center and scale with respect to 200 pixels. Next, it will call 8 stacked hourglass model to detect 2D pose. The detected 2D pose will be fed into siamese network and then output 3D pose. 

![image-20201220101154329](https://raw.githubusercontent.com/skaudrey/picBed/master/img/20201220101213.png)  


## Models
### YOLO
* Check [here](https://pjreddie.com/darknet/yolo/) for more details of model.
* Input: captured image, either single or multiple people.
* Output: the center and scale with respect to 200 pixels (a preparation for hourglass model) according to YOLO's prediction-- bounding boxes. ```c: (#people,2); s: (#people,1)```
### Hourglass-pytorch
* Check paper for more details. 
* Input: captured image which is resized to (256,256,3), ```c,s``` by YOLO.
* Output: 2D poses in shape ```(#people,16,2)```. The joint order is ```[RHip, RKnee, RFoot, LHip, LKnee, LFoot, Hip, Thorax, Neck, Head, LShoulder, LElbow, LWrist, RShoulder, RElbow, RWrist]```

### Siamese-tensorflow
* Computation requirements: Two T4, using augmented cameras and cross-cameras, 100 epochs takes around 9 hours. To train it, check codes [here](https://github.com/vegesm/siamese-pose-estimation).
* Inputs: While training, there are 50% of inputs pairs have the same absolute poses, while the other half part are negative samples.
```Shape = (#poses * 32, #poses* 32,#poses * 3 * 3)```, where the 1st two indicates two 2D poses, the last one is the rotation matrix.
* Output: ```(48,)```, reshaped to ```(16,3)```. The order of the joints is:
```[RHip, RKnee, RFoot, LHip, LKnee, LFoot, Hip, Thorax, Neck, Head, LShoulder, LElbow, LWrist, RShoulder, RElbow, RWrist]```


## Pretrained model
* YOLO V3 trained on COCO: [cfg](https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg), [weights](https://pjreddie.com/media/files/yolov3.weights), [COCO names](https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names)
* 8 stacked hourglass model (pytorch) trained on MPI: http://www-personal.umich.edu/~cnris/original_8hg/checkpoint.pt
* Siamese 3D pose model(trained on Human 3.6m, python 3.6): https://drive.google.com/file/d/1K5a49TLILSFn-JqVsHZNon9Ui0L-BPYh/view?usp=sharing

## Structure:

* ```data/```: dataset and pretrained models
    * ```hg/```: 8 stacked hourglass model, it's pretrained on MPII 
    * ```siamese/```: siamese 3D pose model, it's pretrained on Human3.6m
    * ```yolo/```: yolo v3 model, pretrained on COCO. 
* ```model/```: models used for inducing 3D pose
    * ```hourglass/```:load 8 stacked hourglass model and do 2d pose detection.  
    * ```siamese/```: load siamese 3D pose model and do 3D pose detection.
    * ```yolo/```: load yolo v3 model and predict the center and scale with respect to 200 pixels by each person's bounding box.
* ```utils/```: utilities: plot 2D pose and 3D pose, image preprocessing and post-processing.
* ```main.py```: main script, will start a server of capturing image and predict 3D pose.
* ```client.py```: A demo of client, which will send capture request and receive predicted 3D pose from server.


## Training and Testing
* To start a server, call:

```python main.py -h <HOST> -p <PORT> -all True``` 

Option ```-h, --host``` and ```-p, --port``` allows you to set on which ip and port you're gonna deploy this server. Option ```-all, --all_loaded``` specify whether to load all models before or do it in each prediction. This option depends the capacity of your GPU, if it's powerful, set it as ```True```, otherwise let it be ```False```. When ```-all``` is ```False```, at least one GeForce MX130 is required, even with this you may wait patiently for about 2 minutes to complete all predictions, the main time consumption comes from loading models one by one. 

* To start a client, call
```python client.py -h <HOST> -p <PORT>``` 

Option ```-h, --host``` and ```-p, --port``` allows you to define on which ip and port that the server is running.) To ask for one capture and get 3D pose back, type ```capture``` in client console. Once the server send the prediction result, client will show the prediction result in client console.

## Tips
* Make sure your camera will get your whole body's image, because all models for pose detecting are trained on whole pose. If your camera can only get the upper body's image, then the detected pose will collapse. Model will still try to detect knee, foot mistakenly.
* If you want test predicting 3D pose rather than start a server, you can directly run pred2D.py under ```model/hourglass/```, there are 3 examples offered, given pre-detected bounding boxes.


## Reference
### Papers
* [Stacked Hourglass Networks for Human Pose Estimation](https://arxiv.org/pdf/1603.06937.pdf)
* [From Depth Data to Head Pose Estimation: a Siamese approach](https://arxiv.org/pdf/1703.03624.pdf)
### Codes
* https://github.com/princeton-vl/pytorch_stacked_hourglass
* https://github.com/vegesm/siamese-pose-estimation
* https://github.com/luaffjk/Image-Processing/blob/master/detecting.ipynb
