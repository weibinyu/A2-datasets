a2.clear()
load('GPUbenchmark.csv')

X = GPUbenchmark(:,1:6);
y = GPUbenchmark(:,7);

%%coefficients is Beta
[model,selectedF] = fs(X,y);
kfcv(X,y,3,model,selectedF)

%% k-fold cross-validation
function cVali = kfcv(X,y,k,model,selectedF)
n = length(y);
r = randperm(length(y)); % permutation of the vector 1:n
f = floor(n/k); % largest integer ? n/k
rp = num2cell(reshape(r(1:k*f), f, k), 1); % partition of r into k cell arrays
for i=1:(n-f*k)
rp{i} = [rp{i} ; r(k*f+i)]; % adding the remaining examples
end
for i=1:k
idx = setdiff(cat(1,rp{:}),rp{i});
tmpX = X(idx,:);
tmpy = y(idx);
temp = ones(size(tmpy));
length(model)
pre = [];
for i = 1:length(model)
    cM = model{i};
    B = cM.Coefficients.Estimate;
    temp = [temp, tmpX(:,selectedF(i))];
    tmpX(:,selectedF(i))=[];
end
end
cVali = 0;
end

%% forward selection
function [f,selectedF] = fs(X,y)
p = size(X,2)+1;
XX = ones(size(y));
mdl = fitlm(XX,y);
Model{1} = mdl;
selectedF=[];
for k=1:p-1
    mse = Model{1}.MSE;
    n=1;
    for i=1:p-k
        new_mdl = fitlm([XX,X(:,i)],y);
        if new_mdl.MSE < mse
            mse = new_mdl.MSE;
            best = new_mdl;
            n = i;
        end
    end
    Model{k+1}= best;
    selectedF = [selectedF,n]
    XX = [XX,X(:,n)];
    X(:,n)=[];
end

f = Model;
end











