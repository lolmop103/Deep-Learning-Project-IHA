
% Load the ground truth data
%load('DaataAndCars.mat')
%load('rcnnStopSigns.mat')

%data = read(trainingDataImds);
%preprocessedTrainingData = transform(trainingData, @(data)preprocessData(data,inputSize));
%numAnchors = 3;
%anchorBoxes = estimateAnchorBoxes(preprocessedTrainingData,numAnchors)

net = layerGraph(trainedNetwork_ResNet50);

analyzeNetwork(net);

%Training Data 
%We need to feed ur labels
numClasses = size(trainingData,2)-1;

featureLayer = 'fire5-concat';

Anchors = [
   155   247
   318   176
   176   304];

lgraph = fasterRCNNLayers([416 416 3],numClasses,Anchors,net,featureLayer);

analyzeNetwork(lgraph);

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

    rcnn = trainRCNNObjectDetector(trainingData, lgraph, options, ...
    'NegativeOverlapRange', [0 0.3], 'PositiveOverlapRange',[0.5 1])





