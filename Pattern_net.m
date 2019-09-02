function [acctrain,accval,acctest] = pattern(data,label)
% A pattern recognition network
trainFcn = 'trainscg';  

% Create a Pattern Recognition Network
hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize, trainFcn);

net.trainParam.showWindow = false;
net.divideFcn = 'divideblock';
net.divideParam.trainRatio = .5;
net.divideParam.valRatio = .3;
net.divideParam.testRatio = .2;
net.layers{1}.transferFcn = 'purelin';
net.layers{2}.transferFcn = 'softmax';

t = full(ind2vec(label));
[net1,tr] = train(net,data,t);

y = net1(data);
pred = vec2ind(y);
acctrain = sum((pred(tr.trainInd)-label(tr.trainInd))==0)/length(tr.trainInd)*100;
accval = sum((pred(tr.valInd)-label(tr.valInd))==0)/length(tr.valInd)*100;
acctest = sum((pred(tr.testInd)-label(tr.testInd))==0)/length(tr.testInd)*100;


end

