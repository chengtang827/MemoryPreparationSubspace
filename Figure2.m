clear;
load('dataset_pfc.mat');
loc = 1:7; % number of locations
locsize = 7;
N = size(datasets,1); % number of neurons

%% Two different decorrelations between (D1,D2) and (D1,pre-saccade)

% Decorrelation between D1 and D2 activity
[Mcomp1,Pcomp,a1,b1,minmi1] = Decorrelation(d1mean,d2mean); 
% Decorrelation between D1 and pre-saccade activity
[Mcomp2,Scomp,a2,b2,minmi2] = Decorrelation(d1mean,psmean); 
%% Figure 2a
figure(1);
subplot(1,2,1);

scatter(Mcomp1(:),Mcomp2(:),10,'MarkerEdgeColor',[.7,.7,.7],'MarkerFaceColor',[.7,.7,.7]);
title('Correlation > 0.99');
xticks('');
yticks('');

subplot(1,2,2);
scatter(Pcomp(:),Scomp(:),10,'MarkerEdgeColor',[.7,.7,.7],'MarkerFaceColor',[.7,.7,.7]);
title('Correlation = 0.62');
xticks('');
yticks('');
%% Figure 2b

corrm1s = zeros(N,2); % correlation between m1 and sac
corrps = zeros(N,2); % correlation between prp and sac
for n = 1:N
    [r1,p1] = corrcoef(Mcomp1(n,:),Scomp(n,:));
    corrm1s(n,1) = r1(1,2);
    corrm1s(n,2) = p1(1,2);

    [r2,p2] = corrcoef(Pcomp(n,:),Scomp(n,:));
    corrps(n,1) = r2(1,2);
    corrps(n,2) = p2(1,2);
end
c1 = sum(corrps(ind_sel,2)<0.05)/length(ind_sel);
c2 = sum(corrm1s(ind_sel,2)<0.05)/length(ind_sel);

figure(2);
bar([c1,c2],0.5);
xlabel('correlation')
xticks([1,2]);
xticklabels({'M and S','P and S'});

ylabel('% of cells significantly correlated');
yticks([0,0.8]);
yticklabels({'0','80'});
ylim([0,0.8]);
%% Figure 2c
[~,I1] = sort(corrps(:,1),'descend');
x = 0:2/7*pi:2*pi;
list = [2,9,10,15];% [138 90 134 91]
figure(3);
for i = 1:4
    n = I1(list(i));
    y1 = Pcomp(n,:);
    y2 = Scomp(n,:);
    subplot(2,2,i);
    polarplot(x,[y1 y1(1)]);
    hold on;
    polarplot(x,[y2 y2(1)]);
end
legend('P (preparation)','S (saccade)');