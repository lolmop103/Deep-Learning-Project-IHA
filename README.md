# Deep-Learning-Project-IHA

## Introduction

<!---TO DO-->

## Components

### Images

It was decided to train with 3 different cans: Coca-Cola, Fanta Orange and San Pellegrino Clementina. Of these cans pictures were taken (in .jpg format) from all sides and in different angles. There are between 260-300 pictures per can. These are the pictures that will be used to train the model.
Furthermore, there are two extra cans added to the image set which can be used to test the model, to see how it reacts on cans that were not included in the training.

### Model

#### YOLO

For object detection using deep learning there are three widely used methods. Faster RCNN, YOLO and SSD. We decided to use YOLO as it is the fastest of the three methods. The downside with it being faster is that it loses some accuracy. For the training of the model we used a edited version of the code from the following guides
[preprocessing guide](https://nl.mathworks.com/videos/data-preprocessing-for-deep-learning-1578031296056.html)
[training guide](https://nl.mathworks.com/videos/design-and-train-a-yolov2-network-in-matlab-1578033233204.html)

To train the YOLO model we first had to label all of our data, we did this using the imageLabeler tool of matlab. In this tool we add the different labels: cola, fanta, sanPelligrino and then draw a bounding box around the object (can also be multiple objects) that can be found in the image. When finished the labeled data is saved as a groundTruth. At the end of the preprocessing we then convert this to a table that can be used as input for the training of our model.

For our training we changed the ... to adam, lowered the learning rate to 0.0001 and changed the mini-batch size to 8. Before training it is also important to change the anchors variable, this can be done using the AnchorBoxes file.

# ResNet50

## Reasoning behind model choice

![alt text](./img_read/overall_models.png "image Title")

Had first chosen for squezeNet because it is one of the easiest models to mount and appy transfer learning.

## SqueezeNet

After I was comfortable with it and obtained a tolerable result I started working with ResNet50, I applied the same layers to the output on ResNet50.

## Using ResNet50

ResNet-50 is a convolutional neural network that is trained on more than a million images from the ImageNet database. As a result, the network has learned rich feature representations for a wide range of images. The network can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals.

ResNet-50 is a 50-layer convolutional neural network (48 convolutional layers, one MaxPool layer, and one average pool layer). Residual neural networks are a type of artificial neural network (ANN) that forms networks by stacking residual blocks.

In Mathlab it has 177 layers in total, corresponding to a 50 layer residual network.

The network has an image input size of 224-by-224-by-3.

## Benefits of ResNet50

1.Networks with large number (even thousands) of layers can be trained easily without increasing the training error percentage.

2.ResNets help in tackling the vanishing gradient problem using identity mapping.

The ResNet architecture follows two basic design rules. First, the number of filters in each layer is the same depending on the size of the output feature map. Second, if the feature map’s size is halved, it has double the number of filters to maintain the time complexity of each layer.

### layers of ResNet50

The 50-layer ResNet architecture includes the following elements, as shown in the table below:

|Architecture    |  
|---|
|A 7×7 kernel convolution alongside 64 other kernels with a 2-sized stride.   |
|A max pooling layer with a 2-sized stride.  |  
|9 more layers—3×3,64 kernel convolution, another with 1×1,64 kernels, and a third with 1×1,256   |
|kernels. These 3 layers are repeated 3 times.   |
|12 more layers with 1×1,128 kernels, 3×3,128 kernels, and 1×1,512 kernels, iterated 4 times.   |
|18 more layers with 1×1,256 cores, and 2 cores 3×3,256 and 1×1,1024, iterated 6 times.   |
|9 more layers with 1×1,512 cores, 3×3,512 cores, and 1×1,2048 cores iterated 3 times.   |
| Average pooling, followed by a fully connected layer with 1000 nodes, using the softmax activation function.  |

![alt text](./img_read/architecture-resnet50.webp "image Title")

The first thing I tested to start adjusting the network was to start adjusting the last three layers. The fully connected and class output where sufficient. Placed both on six outputs. Needed to remove the layers to be able to ajust the parameters.

![alt text](./img_read/last_layers_model.gif "image Title")

Inside of ModelTransferLearning folder:

|Files   | Function  |
|---|---|
| test_model.m  | In this file we test the trained CNN  |  
| Adapt_for_cnn.m  | In this folder we change ur model to make it compatible for fast RCNN   |
| resNet_trained_1.mat |The model with all     |

## Data augementation

I tested the trained model with the 6 classes, it was almost a success, the cola can is recognised less,my reasoning was a  larger learning rate. There is an overfitting with the cola bottle over the cola Can. Letting it train for less time, drop out increase....  So have adjusted the model 0.0001 lr.
Got a validation dataset, that i select by myself with lots of noise and odd angles. It seems to work now when held in front of the camera in real time. Not all classes have the same number of pictures and it is not necessary, I also applied data augmentation.

![alt text](./img_read/Data%20.gif "image Title")

## training ResNet50

Many times I did not have to train. I noticed with a low number of epochs 10. A miniBatchSize of 11 and a validation frequency of 6 obtained good results. I chose for a rotation between -90 degrees and 90 degrees. And a max scaling of 2.

![alt text](./img_read/HyperParameters.gif "image Title")

![alt text](./img_read/Reslutofalliterationstraining.gif "image Title")

## Testing

For the moment it reacts better with an black background. The parameters i'm using to train the model are the next.

|parameters  | Function  | Value   |
|---|---|---|
| adam  | Main function for the training process  |   |
| learning rate|How fast we converagance to an result |0.001    |
| mean/average |The noise function  |  |

## Link to dataset of images

https://vivesonline-my.sharepoint.com/:f:/r/personal/r0755466_student_vives_be/Documents/Data-foto%27s-deepl/NewDataset?csf=1&web=1&e=SQyhZf

## Needed toolboxes (Add ons)

Deep Learning Toolbox
Deep Learning Toolbox model for ResNet50 network

## Bibliografie
https://datagen.tech/guides/computer-vision/resnet-50/#:~:text=The%2050%2Dlayer%20ResNet%20architecture,with%201%C3%971%2C256%20kernels.

https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec

https://tensorspace.org/html/playground/resnet50.html

https://iq.opengenus.org/resnet50-architecture/

https://iq.opengenus.org/residual-neural-networks/#:~:text=Advantages%20of%20ResNet,gradient%20problem%20using%20identity%20mapping.

### GUI

<!---TO DO-->

### Database

For the database there wasn't enough time to create one. So we made a json-file that consists of health information about how much sugar and caffeine an adult or a child can consume.
Then there is also a way to access the can with a specific ID, like "coca-cola". This ID has the ingredients of that specific can and how much of it you can drink, until you hit the limit of sugar and caffeine.

The information about health was found on [gezondleven.be](https://www.gezondleven.be/themas/voeding/voedingsdriehoek) and the ingredients were found on the cans themselves.
