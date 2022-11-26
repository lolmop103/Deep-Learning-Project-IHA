
% we load the saved workspace 
load('resNet_trained_1.mat')
% we make an instance to use the trained model 
net = trainedNetwork_ResNet50;

%227 227 
inputSize = net.Layers(1).InputSize(1:2);

camera = webcam;

h = figure;

while ishandle(h)
    im = snapshot(camera);
    image(im)
    im = imresize(im,inputSize);
    [label,score] = classify(net,im);
    title({char(label), num2str(max(score),2)});
    drawnow
end




