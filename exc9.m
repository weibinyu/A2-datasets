a2.clear();
load('microchiptests.csv');
X1 = microchiptests(:,1);
X2 = microchiptests(:,2);
y = microchiptests(:,3);

error = ones(9,1);
errorU = ones(9,1);
cross = ones(9,1);
crossU = ones(9,1);
for i=1:9
d = i;
a = 0.01;
L = 5;
B = initB(d);
mapped = mapFeature(X1,X2,d);
[itera,beta,cost] = gradientDescentReg(mapped,y,B,a,L);

subplot(3,3,i)
plot2dContour(beta,mapped,y,d)
title("degree is "+ i)

fun = @(B)((-1)/size(mapped,1))*((y.')*log(a2.sigmoid(mapped*B))+((1-y).')*log(1-a2.sigmoid(mapped*B)));
[theta, final_cost] = fminunc(fun, B);

error(i) = errorCalc(mapped,y,beta);
errorU(i) = errorCalc(mapped,y,theta);

cross(i) = kfcv(mapped,y,3,beta);
crossU(i) = kfcv(mapped,y,3,theta);
end

figure
plot(1:9,error);
hold on;
plot(1:9,errorU);

figure
plot(1:9,cross);
hold on;
plot(1:9,crossU);

%% cost reg
function clr = costLogistReg(X,y,beta,L)
beta_reg = beta(1,1);
clr = ((-1)/size(X,1))*((y.')*log(a2.sigmoid(X*beta))+((1-y).')*log(1-a2.sigmoid(X*beta)))+(L/size(X,1))*((beta'*beta)-beta_reg^2);
end

%% gradient reg
function [itera,beta,cost] = gradientDescentReg(X,y,beta,a,L)
cost = costLogistReg(X,y,beta,L);
itera = 0;
while true
    itera = itera+1;
    beta_reg = beta(2:end);
    grad = (X.')*(a2.sigmoid(X*beta)-y);
    grad_reg = grad + L/(size(X,1))*[0; beta_reg];
    next_beta = beta-(a*grad_reg);
    new_cost = costLogistReg(X,y,next_beta,L);
    beta = next_beta;
    if cost > new_cost
        cost = new_cost;
    else
        break;
    end
end
end

%% mapped X by degree
function out = mapFeature(X1, X2, D)
out = ones(size(X1(:,1)));
for i = 1:D
    for j = 0:i
        out(:, end+1) = (X1.^(i-j)).*(X2.^j);
    end
end
end

%% init B
function beta = initB(d)
B=[];
for n=1:d
B=[B;zeros(3,1)];
end
ex = d-2;
if ex > 0
    ex = linspace(min(1),max(ex),ex);
    ex = sum(ex);
    for i=1:ex
        B=[B;[0;]];
    end
end
beta = B;
end

%% plot desicsion boundary
function plot2dContour(beta,X,y,degree)
gscatter(X(:,2),X(:,3),y,'br','.',8,'on');
hold on
sz = 100;
x0=linspace(min(X(:,2)), max(X(:,2)), sz);
y0=linspace(min(X(:,3)), max(X(:,3)), sz);
z = zeros(length(x0), length(y0));
% Evaluate z = X*beta over the grid
for i = 1:length(x0)
    for j = 1:length(y0)
    z(i,j) = mapFeature(x0(i), y0(j),degree)*beta;
    end
end
z = z'; % important to transpose z before calling contour
% Plot z = 0. Notice you need to specify the range [0, 0]
contour(x0, y0, z, [0, 0], 'LineWidth', 2)

end

%% error calculation
function err = errorCalc(X,y,beta)
    y1 = a2.sigmoid(X*beta);
    y1 = round(y1);
    err = sum(y1~=y);
end

%% cross validation.
function crossV = kfcv(X,y,k,model)
errors = 0;
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

errors = errors + errorCalc(tmpX,tmpy,model);
end
crossV = errors/k;
end
