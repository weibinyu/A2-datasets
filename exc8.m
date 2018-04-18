a2.clear()
load('GPUbenchmark.csv')

X = GPUbenchmark(:,1:6);
y = GPUbenchmark(:,7);
kfcv(X,y,2)

%% k-fold cross-validation
function cVali = kfcv(X,y,k)
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

%train model on tmpX, tmpy

end
cVali = 0;
end

%% forward selection
