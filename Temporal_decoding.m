function [performance,acc_loc] = ...
    Temporal_decoding(dataset,m,st,session,Trial_Label,trials,bins,~,~,~,~,Test,loc,space)


% Builds a set of training and testing trial labels to build the
% training and testing dataset that will eventually be fed to the decoder.
[train_trials,test_trials,training_label,testing_label] = MakeTrialSet(trials,session,Test,loc,Trial_Label);

% Initializing the training dataset with zeros
train_data = zeros(size(dataset,1),size(train_trials,2),size(dataset,3));
% Intializing the testing dataset with zeros
test_data = zeros(size(dataset,1),size(test_trials,2),size(dataset,3));
% Filling up train_data from dataset. All the values are z_scored using m,st to normalize the firing rate across neurons
for n = 1:size(dataset,1)
    train_data(n,:,:) = ((dataset(n,train_trials(n,:),:))-m(n));
    test_data(n,:,:) = ((dataset(n,test_trials(n,:),:))-m(n))
end

acc_loc = struct;
% Looping through the training time bins
for i_b = 1:size(bins,2)-1
    % Looping through the testing time bins
    for i_bins = 1:size(bins,2)-1
        % De-noising the dataset to feed into the decoder
        [train_data_new,test_data_new] = Build_DataSet(squeeze(train_data(:,:,i_b)),squeeze(test_data(:,:,i_bins)),space);
%                 train_data_new = squeeze(train_data(:,:,i_b))'*space;
%                 test_data_new = squeeze(test_data(:,:,i_bins))'*space;
        %         test_data_new = squeeze(train_data(:,:,i_bins))'*space;
        [performance(i_b,i_bins) acc_loc(i_b, i_bins).val] = ComputePerformance(train_data_new,test_data_new,training_label,testing_label);
    end
end
end
%% Function to create a matrix of training and testing labels
function [train_trials,test_trials,train_labels,test_labels] = MakeTrialSet(trials,session,Test,loc,Trial_Label)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%need to take care of location label as well
flag = 1;% correct mode
if strcmp(Test,'error')
    flag = 0; % error mode
    loc = [2 3 5 6];
end
N = size(session,1);
ss = 1000;
train_trials = zeros(N,ss);
test_trials = zeros(N,ss);
train_labels = randsample(loc,ss,'true');
test_labels = randsample(loc,ss,'true');
for n = 1:N
    label = AssignTrialLabel(trials(session(n)).val,1);
    label = label';
    cor_ind = find(label>0);
    err_ind = find(label<0);
    train_set = randsample(cor_ind,...
        round(0.50*(length(cor_ind))));
    if flag==1
        test_set = setdiff(cor_ind,train_set);
    else
        test_set = err_ind;
    end
    train_set = [train_set label(train_set)];
    if flag==1
        test_set = [test_set label(test_set)];
    else
        test_set = [test_set -label(test_set)];
    end
    for i = 1:length(train_labels)
        train_trials(n,i) = randsample(train_set(train_set(:,2)==train_labels(i),1),1);
        test_trials(n,i) = randsample(test_set(test_set(:,2)==test_labels(i),1),1);
    end
end
end

%% Function to build the training and testing set
% This function denoises the train and test dataset using PCA. The data
% used to decode is the data projected onto the principal components.
function [train_data_new,test_data_new] = Build_DataSet(train_data,test_data,space)
% Initializing a matrix to store all the neuron indices with NaN values.
% These are neurons with very low firing.
train_data = space'*train_data;
test_data = space'*test_data;
ind_nan=[];
% Checking for neurons with NaN values within the train data
for i = 1:size(train_data,1)
    if ~isfinite(train_data(i,1))
        ind_nan = [ind_nan i];
    end
end
% Checking for neurons with NaN values within the test data
for i = 1:size(test_data,1)
    if ~isfinite(test_data(i,1))
        ind_nan = [ind_nan i];
    end
end
% Pick all the neurons that had NaNs in the train and/or test data.
ind_nan = unique(ind_nan);
% Getting rid of these neurons in the train and test dataset.
train_data(ind_nan,:)=[];test_data(ind_nan,:)=[];
% Creating a PCA space with training and testing data.
A = [train_data';test_data'] - mean([train_data';test_data'],2);

[V,D] = eig(cov(A));
%[coeff,score,latent] = princomp([train_data';test_data']);
score = A*fliplr(V);
% Computing the proportion of explained variance for each component
latent = sort(diag(D),'descend');
latent = cumsum(latent);
latent = latent/latent(end);
% Finding the number of components explaining 90% of the variance.
if space==1
    expl_var = dsearchn(latent,0.95);
elseif size(space,1)==size(space,2)
    expl_var = dsearchn(latent,0.999);
elseif size(space,2)==1
    expl_var = 1;
else
    expl_var = size(space,2)-1;
end

train_data_new = score(1:size(train_data,2),1:expl_var);
% Denoising the test data similarly.
test_data_new = score(size(train_data,2)+1:end,1:expl_var);

end
%% Calculating decoding performance using an LDA
function [performance, acc_loc] = ComputePerformance(train_data,test_data,training_label,testing_label)
% class is the predicted target label from the test_data
[pred,err,posterior,logp,coef] = classify(test_data,train_data,training_label);
% Checking how different the predicted label is from the actual label for
% all the test trials
perf = pred-testing_label';
labels = unique(testing_label);
acc_loc = zeros(length(labels),1);
for i = 1:size(acc_loc,1)
    label = labels(i);
    acc_loc(i) = sum(pred(testing_label==label)==label)/sum(testing_label==label);
end

% Extracting all the correctly predicted target label
perf_ind = find(perf==0);
% The performance of the decoder is computed as the percentege of number of correct
% predictions in the decoding.
performance = length(find(perf==0))*100/length(testing_label);
end
%%
