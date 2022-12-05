# Deep-Learning-Project-IHA

## Introduction

<!---TO DO-->

## Add-ons matlab

These are the add-ons you need:

- Parallel computing toolbox
- MATLAB Support Package for USB Webcams
- Computer Vision Toolbox
- Image Processing Toolbox
- Deep Learning Toolbox

### Add-on installation

To install the parallel computing toolbox, you'll will probably have problems finding the add-on. For this problem, you need to search for "parallel computing toolbox" and click on another add-on. There you will see that you can install the parallel computing toolbox that you need.

![parallel computing](./img/documentation/parallel_computing_problem.png)

You could probably have the same problem with the "computer vision toolbox", therefore you can search for "computer vision toolbox", and click on another add-on.

![computer vision](./img/documentation/computer_vision_problem.png)

If you installed every add-on then your installed add-on window will look like this.

![add-on window](./img/documentation/add_on_manager.png)

## Components

### Images

It was decided to train with 3 different cans: Coca-Cola, Fanta Orange and San Pellegrino Clementina. Of these cans pictures were taken (in .jpg format) from all sides and in different angles. There are between 260-300 pictures per can. These are the pictures that will be used to train the model.
Furthermore, there are two extra cans added to the image set which can be used to test the model, to see how it reacts on cans that were not included in the training.

### Model

#### YOLO

For object detection using deep learning there are three widely used methods. Faster RCNN, YOLO and SSD. We decided to use YOLO as it is the fastest of the three methods. The downside with it being faster is that it loses some accuracy. For the training of the model we used a edited version of the code from the following guides

- [preprocessing guide](https://nl.mathworks.com/videos/data-preprocessing-for-deep-learning-1578031296056.html)
- [training guide](https://nl.mathworks.com/videos/design-and-train-a-yolov2-network-in-matlab-1578033233204.html)

To train the YOLO model we first had to label all of our data, we did this using the imageLabeler tool of matlab. In this tool we add the different labels: cola, fanta, sanPelligrino and then draw a bounding box around the object (can also be multiple objects) that can be found in the image. When finished the labeled data is saved as a groundTruth. At the end of the preprocessing we then convert this to a table that can be used as input for the training of our model.

For our training we changed the ... to adam, lowered the learning rate to 0.0001 and changed the mini-batch size to 8. Before training it is also important to change the anchors variable, this can be done using the AnchorBoxes file.

#### What is YOLO

YOLO is a real-time object detection algorithm. The authors of YOLO frame the object detection problem as a regression problem instead of a classification task by spatially separating bounding boxes and associating probabilities to each of the detected images using a single convolutional neural network (CNN).

##### Benefits of YOLO

- speed
- Detection accuracy
- lots of examples/documentation

We have chosen YOLO because it is a very fast object-detection algorithm that still has a pretty good detection accuracy. But most importantly we choose YOLO because it is widely used and documented. This means we could find way more examples then for e.g. RCNN.  
![object detection speeds](./img/documentation/speed.png)

##### YOLO architecture

![YOLO architecture](./img/documentation/YOLO_architecture.png)

- Resizes the input image into 448x448 before going through the convolutional network. The size of the input image can however be changed, the example code we used to make our model uses a input size of 128x128.
- A 1x1 convolution is first applied to reduce the number of channels, which is then followed by a 3x3 convolution to generate a cuboidal output.
- The activation function under the hood is ReLU, except for the final layer, which uses a linear activation function.
- Some additional techniques, such as batch normalization and dropout, respectively regularize the model and prevent it from overfitting.

#### How does the YOLO algorithm work

1. Residual blocks  
First the input image is divided into a grid of NxN. Each cell in the grid then has localize and predict the class of the object it covers along with a probability value.
2. Bounding box regression  
Next the bounding boxes that correspond to objects in the image need to be determined. YOLO determines the attributes of these bounding boxes using a single regression module in the following format, where Y is the final vector representation for each bounding box.
Y = [pc, bx, by, bh, bw, c1, c2]

- pc corresponds to the probability score of each grid that contains an object.  
![probability score of each grid](./img/documentation/probability_score.png)
- bx and by are the x and y coordinates of the center of the bounding box with respect to the enveloping grid cell.
- bh, bw correspond to the height and the width of the bounding box with respect to the enveloping grid cell.
- c1 and c2 correspond to the classes we are trying to detect. You can have as many classes as your use case requires.  
![bounding boxes](./img/documentation/bounding_boxes.png)

3. intersections over Unions or IOU  
In YOLO an object in an image can have multiple grid box candidates. The goal of the IOU is to discard grid boxes that are not relevant.
First the user has to define an IOU threshold which decides how high a prediction of a grid box has to be for it to be relevant. YOLO computes the IOU of each grid cell which is the Intersection area divided by the Union Area.
![IOU](./img/documentation/IOU.png)
4. Non-Max Suppression or NMS  
Setting a threshold for the IOU is not always enough because an object can have multiple boxes with IOU beyond the threshold.  
This is where NMS can be used to only keep the boxes with the highest probability score of detection.

### GUI

<!---TO DO-->

### Database

For the database there wasn't enough time to create one. So we made a json-file that consists of health information about how much sugar and caffeine an adult or a child can consume.
Then there is also a way to access the can with a specific ID, like "coca-cola". This ID has the ingredients of that specific can and how much of it you can drink, until you hit the limit of sugar and caffeine.

The information about health was found on [gezondleven.be](https://www.gezondleven.be/themas/voeding/voedingsdriehoek) and the ingredients were found on the cans themselves.
