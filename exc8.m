a2.clear()
load('GPUbenchmark.csv')

X = GPUbenchmark(:,1:6);
y = GPUbenchmark(:,7);

[model,selectedF] = fs(X,y);
best = kfcv(X,y,3,model,selectedF);
best1 = AIC(X,model);

%% AIC
function aic = AIC(X,model)
n = length(X);
C = log(2*3.14) + 1;
A = [];
for i = 1: length(model)
    A(i) = n*log(model{i}.MSE)+2*(i+1)+(n*C);
    if i == 1
        best = A(i);
        bestM = i;
    elseif best > A(i)
        best = A(i);
        bestM = i;
    end
end
aic = bestM;
end

%% k-fold cross-validation
function cVali = kfcv(X,y,k,model,selectedF)
n = length(y);
r = randperm(length(y)); % permutation of the vector 1:n
f = floor(n/k); % largest integer ? n/k
rp = num2cell(reshape(r(1:k*f), f, k), 1); % partition of r into k cell arrays

for i=1:(n-f*k)
rp{i} = [rp{i} ; r(k*f+i)]; % adding the remaining examples
end
mse = [ones(size(model,2),1)];
for i=1:k
idx = setdiff(cat(1,rp{:}),rp{i});
tmpX = X(idx,:);
tmpy = y(idx);
temp = ones(size(tmpy));

for n = 2:length(model)
    cM = model{n};
    B = cM.Coefficients.Estimate;
    B(1,:) = [];
    temp = [temp, tmpX(:,selectedF(n-1))];
    tmpX(:,selectedF(n-1))=[];
    mse(n,i) = J(temp,tmpy,B);
end
mse = [mse,ones(size(model,2),1)];

end
mse(1,:) = [];
best = BestM(mse);
for nn =1:length(best)
    %%add one because i removed M0 before since M0 is absolutly not the best.
    disp("For "+nn+"'s set of sub data, model "+(best(nn)+1)+" was best");
end
cVali = best;
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
    selectedF = [selectedF,n];
    XX = [XX,X(:,n)];
    X(:,n)=[];
end

f = Model;
end

function pre = predict(X,B)
pre = X*B;
end

function mse = J(X,y,B)
mse = (X*B-y)'*(X*B-y);
end

function bestM = BestM(mse)
best = ones(size(mse,2)-1,1);
bestM = ones(size(mse,2)-1,1);
for i =1:size(mse,2)-1
    best(i,1) = mse(1,1);
    for n =1:size(mse,1)
        if best(i)> mse(n,i)
            best(i) = mse(n,i);
            bestM(i) = n;
        end
    end
end
end
