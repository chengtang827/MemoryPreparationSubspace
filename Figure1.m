clear;
load('dataset_pfc.mat');
loc = 1:7; % number of locations
locsize = 7;
N = size(datasets,1); % number of neurons
%% Cross-temporal decoding in the full space, Figure 1 b
run = 1;
window = 58;
perf_full = zeros(window-1,window-1,run);
for it = 1:run
    [perf_full(:,:,it),~] = Temporal_decoding(datasets(:,:,1:window), m,st,...
        sessions,'target' ,trials,bins(:,1:window), [], [], [], [], 'correct',loc,1);
    display(it);
end
figure(1); 
imagesc(flipud(mean(perf_full(1:window-1,1:window-1,:),3)),[100/length(loc),45]);
colormap('jet');
colorbar;
xticks('');
yticks('');
lw = 2;
line([6,6],[0,window],'color','w','linewidth',lw);
line([12,12],[0,window],'color','w','linewidth',lw);
line([32,32],[0,window],'color','w','linewidth',lw);
line([38,38],[0,window],'color','w','linewidth',lw);
line([0,window],[window-6,window-6],'color','w','linewidth',lw);
line([0,window],[window-12,window-12],'color','w','linewidth',lw);
line([0,window],[window-32,window-32],'color','w','linewidth',lw);
line([0,window],[window-38,window-38],'color','w','linewidth',lw);
%% Identify working memory and motor preparation subspaces by decorrelating
[Mcomp,Pcomp,a,b,minmi] = Decorrelation(d1mean,d2mean); % Decorrelate and 
% find components with minimal mutual information
Mspace = gramschmidt(Mcomp,Mcomp(:,1)); % find orthogonal basis
Pspace = gramschmidt(Pcomp,Pcomp(:,1));
%% Cross-temporal decoding in the working memory subspace, Figure 1d
run = 1;
window = 58;
perf_m = zeros(window-1,window-1,run);
for it = 1:run
    [perf_m(:,:,it),~] = Temporal_decoding(datasets(:,:,1:window), m,st,...
        sessions,'target' ,trials,bins(:,1:window), [], [], [], [], 'correct',loc,Mspace);
    display(it);
end
figure(2); 
imagesc(flipud(mean(perf_m(1:window-1,1:window-1,:),3)),[100/length(loc),45]);
colormap('jet');
colorbar;
xticks('');
yticks('');
lw = 2;
line([6,6],[0,window],'color','w','linewidth',lw);
line([12,12],[0,window],'color','w','linewidth',lw);
line([32,32],[0,window],'color','w','linewidth',lw);
line([38,38],[0,window],'color','w','linewidth',lw);
line([0,window],[window-6,window-6],'color','w','linewidth',lw);
line([0,window],[window-12,window-12],'color','w','linewidth',lw);
line([0,window],[window-32,window-32],'color','w','linewidth',lw);
line([0,window],[window-38,window-38],'color','w','linewidth',lw);
%% Cross-temporal decoding in the motor preparation subspace, Figure 1e
run = 1;
window = 58;
perf_p = zeros(window-1,window-1,run);
for it = 1:run
    [perf_p(:,:,it),~] = Temporal_decoding(datasets(:,:,1:window), m,st,...
        sessions,'target' ,trials,bins(:,1:window), [], [], [], [], 'correct',loc,Pspace);
    display(it);
end
figure(3); 
imagesc(flipud(mean(perf_p(1:window-1,1:window-1,:),3)),[100/length(loc),45]);
colormap('jet');
colorbar;
xticks('');
yticks('');
lw = 2;
line([6,6],[0,window],'color','w','linewidth',lw);
line([12,12],[0,window],'color','w','linewidth',lw);
line([32,32],[0,window],'color','w','linewidth',lw);
line([38,38],[0,window],'color','w','linewidth',lw);
line([0,window],[window-6,window-6],'color','w','linewidth',lw);
line([0,window],[window-12,window-12],'color','w','linewidth',lw);
line([0,window],[window-32,window-32],'color','w','linewidth',lw);
line([0,window],[window-38,window-38],'color','w','linewidth',lw);
%% Visualization of the 2 subspaces in top 3 PCs, Figure 1f

points = 50; % number of points shown for each target location
ss = 25; % number of trials that are averaged for each point

mem1 = zeros(N,locsize,points); % memory in Delay 1
mem2 = zeros(N,locsize,points); % memory in Delay 2

prp1 = zeros(N,locsize,points); % preparation in Delay 1
prp2 = zeros(N,locsize,points); % preparation in Delay 2

dl1_point = zeros(N,locsize,points); % Delay 1 activity for each point
dl2_point = zeros(N,locsize,points); % Delay 2 activity for each point
for n = 1:N
    label = AssignTrialLabel(trials(sessions(n)).val,1);
    for tar = loc
        t_list = find(label==tar);
        for p = 1:points
            t_num = randsample(t_list,ss,'true');
            d1_t = mean(mean(datasets(n,t_num,21:31)))-m(n);
            t_num = randsample(t_list,ss,'true');
            d2_t = mean(mean(datasets(n,t_num,43:53)))-m(n);           
            dl1_point(n,tar,p) = d1_t;
            dl2_point(n,tar,p) = d2_t;
            mem1(n,tar,p) = d1_t - a*Pcomp(n,tar);
            mem2(n,tar,p) = d2_t - Pcomp(n,tar);
            prp1(n,tar,p) = d1_t - Mcomp(n,tar);
            prp2(n,tar,p) = d2_t - b*Mcomp(n,tar);            
        end
    end
end

m1f = reshape(mem1,size(mem1,1),locsize*points);
m2f = reshape(mem2,size(mem2,1),locsize*points);
p1f = reshape(prp1,size(prp1,1),locsize*points);
p2f = reshape(prp2,size(prp2,1),locsize*points);
d1f = reshape(dl1_point,size(dl1_point,1),locsize*points);
d2f = reshape(dl2_point,size(dl2_point,1),locsize*points);


mpdata = [m1f m2f p1f p2f];
d12data = [d1f d2f];

mpdata = mpdata-mean(mpdata,2);
d12data = d12data-mean(d12data,2);

m_comp_sub_data = Mspace'*mpdata;
p_comp_sub_data = Pspace'*mpdata;
m_sub_data = Mspace'*d12data;
p_sub_data = Pspace'*d12data;

m_comp_sub_data = m_comp_sub_data-mean(m_comp_sub_data,2);
p_comp_sub_data = p_comp_sub_data-mean(p_comp_sub_data,2);
m_sub_data = m_sub_data-mean(m_sub_data,2);
p_sub_data = p_sub_data-mean(p_sub_data,2);

[V_m_comp_sub,~] = eig(cov(m_comp_sub_data'));
[V_p_comp_sub,~] = eig(cov(p_comp_sub_data'));
[V_m_sub,~] = eig(cov(m_sub_data'));
[V_p_sub,~] = eig(cov(p_sub_data'));
V_m_comp_sub2 = V_m_comp_sub(:,end-1:end)'*Mspace';
V_p_comp_sub2 = V_p_comp_sub(:,end-1:end)'*Pspace';
V_m_sub2 = V_m_sub(:,end-1:end)'*Mspace';
V_p_sub2 = V_p_sub(:,end-1:end)'*Pspace';

[V_mp_comp,~] = eig(cov(mpdata'));
V_mp_comp3 = V_mp_comp(:,end-2:end);
[V_d12,~] = eig(cov(d12data'));
V_d12_3 = V_d12(:,end-2:end);

V_m_comp_2to3 = V_m_comp_sub2*V_mp_comp3;
V_p_comp_2to3 = V_p_comp_sub2*V_mp_comp3;
V_m_2to3 = V_m_sub2*V_d12_3;
V_p_2to3 = V_p_sub2*V_d12_3;


ndim3 = 3;
locxss = locsize*points;
%
m1_comp_sub3 = reshape(V_m_comp_2to3'*V_m_comp_sub2*mpdata(:,1:locxss),ndim3,locsize,points);
m2_comp_sub3 = reshape(V_m_comp_2to3'*V_m_comp_sub2*mpdata(:,locxss+1:locxss*2),ndim3,locsize,points);
p1_comp_sub3 = reshape(V_p_comp_2to3'*V_p_comp_sub2*mpdata(:,locxss*2+1:locxss*3),ndim3,locsize,points);
p2_comp_sub3 = reshape(V_p_comp_2to3'*V_p_comp_sub2*mpdata(:,locxss*3+1:end),ndim3,locsize,points);
m1_sub3 = reshape(V_m_2to3'*V_m_sub2*d12data(:,1:locxss),ndim3,locsize,points);
m2_sub3 = reshape(V_m_2to3'*V_m_sub2*d12data(:,locxss+1:end),ndim3,locsize,points);
p1_sub3 = reshape(V_p_2to3'*V_p_sub2*d12data(:,1:locxss),ndim3,locsize,points);
p2_sub3 = reshape(V_p_2to3'*V_p_sub2*d12data(:,locxss+1:end),ndim3,locsize,points);
d1_sub3 = reshape(V_d12_3'*d12data(:,1:locxss),ndim3,locsize,points);
d2_sub3 = reshape(V_d12_3'*d12data(:,locxss+1:end),ndim3,locsize,points);

color = [0,0,0;0,0,.8;0,.8,0;0,.8,.8;.8,0,0;.8,0,.8;.8,.8,0;1,1,1];

figure(4);
hold on;
for tar = [3,6]%1:7
    scatter3(m1_sub3(1,tar,:),m1_sub3(2,tar,:),m1_sub3(3,tar,:),500,color(tar,:),'.');
    scatter3(m2_sub3(1,tar,:),m2_sub3(2,tar,:),m2_sub3(3,tar,:),50,'o','MarkerEdgeColor',color(tar,:),...
        'linewidth',1,'MarkerEdgeAlpha',.3);
    
    scatter3(p1_sub3(1,tar,:),p1_sub3(2,tar,:),p1_sub3(3,tar,:),500,color(tar,:),'.');
    scatter3(p2_sub3(1,tar,:),p2_sub3(2,tar,:),p2_sub3(3,tar,:),50,'o','MarkerEdgeColor',color(tar,:),...
        'linewidth',1,'MarkerEdgeAlpha',.3);
    
    
    scatter3(d1_sub3(1,tar,:),d1_sub3(2,tar,:),d1_sub3(3,tar,:),500,color(tar,:),'.');
    scatter3(d2_sub3(1,tar,:),d2_sub3(2,tar,:),d2_sub3(3,tar,:),50,'o','MarkerEdgeColor',color(tar,:),...
        'linewidth',1,'MarkerEdgeAlpha',.3);
end

Mspace3 = gramschmidt(V_m_2to3',V_m_2to3(1,:)');
Pspace3 = gramschmidt(V_p_2to3',V_p_2to3(1,:)');

sample = [1,0;0,1;-1,0;0,-1]*100;
mplane = sample*Mspace3';
pplane = sample*Pspace3';

fill3(mplane(:,1),mplane(:,2),mplane(:,3),'r','facealpha',0.1,'edgealpha',0.2);
fill3(pplane(:,1),pplane(:,2),pplane(:,3),'b','facealpha',0.1,'edgealpha',0.2);


xlabel('PC1');
ylabel('PC2');
zlabel('PC3');
axis equal;

angle = acos(null(Mspace3')'*null(Pspace3'))/pi*180;
grid on;