# Annotations
_Check [here](http://human-pose.mpi-inf.mpg.de/#download) for more details_
* **imgname**: the name of image
    * Shape = \#image*1.
    * String.
    * e.g. ["p1_1.jpg",...]
* **center**: the center coordinates of one's torso box. Similar to the "objpos" in MPII annotations.
    * Shape = \#image*2, 2 because of (x,y) dimension. 
    * Float
* **scale**: person scale w.r.t. 200 px height. 
    * Shape=\#image*1
    * Float
    * E.g., if the height of one person in its raw image is 600 pixels, then scale for him is 3.
    * if dataset is 3Dmaniqui, then scale is around 0.75 cause the height of people is about 150 px.
* **part**: the coordinates of joints. 
    * Shape = \#image*(16*2), if the skeleton model has 16 joints
    * Float
    * Concretely, the joints from index 0 to index 15 indicate: 0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top, 10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist
* **person**: the id of person, 
    * Shape = \#image*1
    * Integer
* **visible**: indicating whether one joint is visible in its corresponding image, 
    * shape = \#image*(16*1)
    * Boolean
* **normalize**: to give a fair evaluation for images with different resolutions, so if images are in the same resolution, normalize are the same
    * shape = \#image*1
    * float
    