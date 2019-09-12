clear;
load('dataset_pfc.mat');
[Mcomp,Pcomp,a1,b1,minmi1] = Decorrelation(d1mean,d2mean); 
loc = 1:7; % number of locations
locsize = 7;
N = size(datasets,1); % number of neurons
%%

%construct the orthogonal basis for each subspace
mem_space = gramschmidt(Mcomp,Mcomp(:,1));
prp_space = gramschmidt(Pcomp,Pcomp(:,1));

con_mem = zeros(N,1);
con_prp = zeros(N,1);

% contribution is the projection length of each neuron in the subspace
for n = 1:N
    con_mem(n) = norm(mem_space(n,:),2);
    con_prp(n) = norm(prp_space(n,:),2);
end

g1 = find(con_mem>con_prp);
g2 = find(con_mem<con_prp);
data1 = con_mem(g1)./con_prp(g1);
data2 = con_prp(g2)./con_mem(g2);
data12 = [2-data2;data1];
indnew = [g2;g1];


bw = 0.21;
thr = 2;

%% Figure 3b
figure(2);
data12norm = (data12-mean(data12))/std(data12);


indprune = find(data12norm>-thr&data12norm<thr);
indleft = indnew(find(data12norm<-thr));
indright = indnew(find(data12norm>thr));
indcenter = indnew(indprune);
hold on;
histogram(data12,'BinWidth',bw,'FaceColor',[.5,.5,.5],'EdgeAlpha',.2);
histogram(data12(data12norm<-thr),'BinWidth',bw,'FaceColor',[.2,.2,1],'EdgeAlpha',.2);
histogram(data12(data12norm>thr),'BinWidth',bw,'FaceColor',[1,.2,.2],'EdgeAlpha',.2);

%%
% boxplot(z);
z1 = data12(find(regional==1));
z2 = data12(find(regional==2));
z3 = data12(find(regional==3));
z4 = data12(find(regional==4));
boxplot([data2box(z1) data2box(z2) data2box(z3) data2box(z4)]);
% p = anova1(con_prp./con_mem,regional,'off');
%%
prune = data12norm(indprune);
prunenorm = (prune-mean(prune))/std(prune);
[h2,p2] = kstest(prunenorm);
title(['P < 10e-3 before pruning, P = ' num2str(p2) ' after pruning']);

linkaxes;
xlim([-4,6]);
xticks(-3:5);
xticklabels([ 5 4 3 2 1 2 3 4 5 ]);
%% Figure 3a
figure(1);
hold on;
scatter(con_mem,con_prp);
scatter(con_mem(indcenter),con_prp(indcenter),50,'MarkerFaceColor',[.5,.5,.5],'MarkerFaceAlpha',1,'MarkerEdgeAlpha',0);

scatter(con_mem(indleft),con_prp(indleft),50,'MarkerFaceColor',[.2,.2,1],'MarkerFaceAlpha',1,'MarkerEdgeAlpha',0);
scatter(con_mem(indright),con_prp(indright),50,'MarkerFaceColor',[1,.2,.2],'MarkerFaceAlpha',1,'MarkerEdgeAlpha',0);
[r,p] = corrcoef(con_mem,con_prp);

xlabel('Contr'' to Memory');
ylabel('Contr'' to Preparation');
title(['r = ' num2str(r(1,2))  ', p < 10e-10']);