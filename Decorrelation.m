function [comp1,comp2,a,b,minmi] = Decorrelation(interval1,interval2)
% Given 2 arrays of activity, find 2 components from them that has the
% least mutual information
arange = 0:0.01:0.99; % range for parameter search
brange = 0:0.01:0.99;

mi = zeros(length(arange),length(brange)); % mutual information
for aind = 1:length(arange)
    for bind = 1:length(brange)
        a = arange(aind);
        b = brange(bind);
        
        x = (b*interval1(:)-interval2(:))/(a*b-1);% preparation component
        y = (a*interval2(:)-interval1(:))/(a*b-1);% memory component
        mi(aind,bind) = computeMI(x,y);
    end
end
[I,J] = find(mi == min(min(mi)));
a = arange(I);
b = brange(J);
minmi = min(min(mi));

comp1 = (interval1-a*interval2)/(1-a*b);
comp2 = (interval2-b*interval1)/(1-a*b);
end


function mi = computeMI(x,y)
binCount = 7;
xgrid = zeros(binCount+1,1);
ygrid = zeros(binCount+1,1);
mi = 0;
binWidth = 1/binCount*100;

for i = 1:binCount+1
    xgrid(i) = prctile(x,min(100,(i-1)*binWidth));
    ygrid(i) = prctile(y,min(100,(i-1)*binWidth));
end


label = zeros(length(x),2);
for i = 1:length(x)
    % for x label
    for j = 1:binCount
        if xgrid(j)<=x(i)&&x(i)<xgrid(j+1)
            label(i,1) = j;
            break;
        end
    end
    % for y label
    for j = 1:binCount
        if ygrid(j)<=y(i)&&y(i)<ygrid(j+1)
            label(i,2) = j;
            break;
        end
    end
end

ss = size(label,1); % sample size

for i = 1:binCount
    for j = 1:binCount
        P_ij = length(find(label(:,1)==i&label(:,2)==j))/ss;
        if P_ij~=0
            P_i = length(find(label(:,1)==i))/ss;
            P_j = length(find(label(:,2)==j))/ss;
            mi = mi + P_ij*log(P_ij/(P_i*P_j));
        end
    end
end


end


