# Deep-Learning-Project-IHA

## Introduction

<!---TO DO-->

## add-ons matlab

- Parallel computing toolbox
- MATLAB Support Package for USB Webcams
- Computer Vision Toolbox
- Image Processing Toolbox
- Deep Learning Toolbox

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
- Good generalization

We have chosen YOLO because it is a very fast object-detection algorithm.
![object detection speeds](./img/documentation/speed.png)

### GUI

<!---TO DO-->

### Database

For the database there wasn't enough time to create one. So we made a json-file that consists of health information about how much sugar and caffeine an adult or a child can consume.
Then there is also a way to access the can with a specific ID, like "coca-cola". This ID has the ingredients of that specific can and how much of it you can drink, until you hit the limit of sugar and caffeine.

The information about health was found on [gezondleven.be](https://www.gezondleven.be/themas/voeding/voedingsdriehoek) and the ingredients were found on the cans themselves.
