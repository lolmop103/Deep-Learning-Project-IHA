%https://nl.mathworks.com/help/vision/ug/faster-r-cnn-examples.html
%We gone convert the network to R-CNN 
% we load the saved workspace 
load('resNet_trained_1.mat')

%I use analyzeNetwork(net) to get the architecure of the trained CNN 

% Load the pretrained model
net = trainedNetwork_ResNet50;

%Before cahnges
analyzeNetwork(net);

lgraph = layerGraph(net);

% Remove the the last 3 layers from ResNet-50. 
layersToRemove = {
    'fc3'
    'fc1000_softmax'
    'classoutput'
    };
lgraph = removeLayers(lgraph, layersToRemove);

% Specify the number of classes the network should classify.
numClasses = 3;
numClassesPlusBackground = numClasses + 1;

% Define new classification layers.
newLayers = [
    fullyConnectedLayer(numClassesPlusBackground, 'Name', 'rcnnFC')
    softmaxLayer('Name', 'rcnnSoftmax')
    classificationLayer('Name', 'rcnnClassification')
    ];

% Add new layers.
lgraph = addLayers(lgraph, newLayers);

% Connect the new layers to the network. 
lgraph = connectLayers(lgraph, 'avg_pool', 'rcnnFC');



%***************************************************
%We Add Bounding Box Regression Layer

% Define the number of outputs of the fully connected layer.
numOutputs = 4 * numClasses;

% Create the box regression layers.
boxRegressionLayers = [
    fullyConnectedLayer(numOutputs,'Name','rcnnBoxFC')
    rcnnBoxRegressionLayer('Name','rcnnBoxDeltas')
    ];

% Add the layers to the network
lgraph = addLayers(lgraph, boxRegressionLayers);

%****************************
%The box regression layers are typically
% connected to same layer the classification branch is connected to.


% Connect the regression layers to the layer named 'avg_pool'.
lgraph = connectLayers(lgraph,'avg_pool','rcnnBoxFC');

% Display the classification and regression branches of Fast R-CNN.
figure
plot(lgraph)
ylim([-5 16])


%******************************
featureExtractionLayer = 'activation_40_relu';

figure
plot(lgraph)
ylim([30 42])

%******************

% Disconnect the layers attached to the selected feature extraction layer.
lgraph = disconnectLayers(lgraph, featureExtractionLayer,'res5a_branch2a');
lgraph = disconnectLayers(lgraph, featureExtractionLayer,'res5a_branch1');

% Add ROI max pooling layer.
outputSize = [14 14]

%*****************************

roiPool = roiMaxPooling2dLayer(outputSize,'Name','roiPool');

lgraph = addLayers(lgraph, roiPool);

% Connect feature extraction layer to ROI max pooling layer.
lgraph = connectLayers(lgraph, 'activation_40_relu','roiPool/in');

% Connect the output of ROI max pool to the disconnected layers from above.
lgraph = connectLayers(lgraph, 'roiPool','res5a_branch2a');
lgraph = connectLayers(lgraph, 'roiPool','res5a_branch1');

% Show the result after adding and connecting the ROI max pooling layer.
figure
plot(lgraph)
ylim([30 42])


%************************

% Add ROI input layer.
roiInput = roiInputLayer('Name','roiInput');
lgraph = addLayers(lgraph, roiInput);

% Connect ROI input layer to the 'roi' input of the ROI max pooling layer.
lgraph = connectLayers(lgraph, 'roiInput','roiPool/roi');

% Show the resulting faster adding and connecting the ROI input layer.
figure
plot(lgraph)
ylim([30 42])

%**************************************************
%If the code runs well, it shows the new added layers for the R-CNN 

%After changes 
analyzeNetwork(net);


%train the network
%numClasses = size(trainingData,2)-1;

%featureLayer = 'activation_40_relu';

%Anchors = [
%   155   247
%   318   176
%   176   304];

%lgraph = fasterRCNNLayers([416 416 3],numClasses,Anchors,net,featureLayer);

%analyzeNetwork(lgraph);

% A trained detector is loaded from disk to save time when running the
% example. Set this flag to true to train the detector.

    % Set training options
    options = trainingOptions('sgdm',...
        'MiniBatchSize', 128, ...
        'InitialLearnRate', 1e-3, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.1, ...
        'LearnRateDropPeriod', 100, ...
        'MaxEpochs', 100, ...
        'Verbose', true);
    
    % Train an R-CNN object detector. This will take several minutes. 

    rcnn = trainRCNNObjectDetector(trainingData, layerGraph(net), options, ...
    'NegativeOverlapRange', [0 0.3], 'PositiveOverlapRange',[0.5 1])


