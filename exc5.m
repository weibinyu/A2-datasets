a2.clear()
data = load("breast-cancer.mat");
data = data.breast_cancer;
data = data(randperm(size(data,1)),:); % Shuffle rows

%% map 2 to 0 and 4 to 1 i think reason is that 0,1 is much easier to understand
%%and 2,4 might cause misunderstanding that there are more than two label.

for i=1:size(data,1)
    if data(i,10) == 2
       data(i,10) = 0;
    else
        data(i,10) = 1;
    end
end

% I have selected 10 set data as test data and 673 set data as training
% reason for this is i am told that more training makes more accurate model
train = data(1:303,:);
test = data(304:683,:);

%% train model and calculate error 
[B,itera,cost] = findB(train(:,1:9),train(:,10),0.005);

trainE = calculateE(train,B);
testE = calculateE(test,B);

%% function
function [beta,itera,cost] = findB(X,y,a)
n = length(X);
XX = [ones(n,1),a2.normalize(X)];
beta = zeros(10,1);
itera = 0;
iterList = [];
costList = [];
cost = calculateC(XX,y,beta);
while true
    itera = itera+1;
    next_beta = beta-(a*(XX.')*(a2.sigmoid(XX*beta)-y));
    new_cost = calculateC(XX,y,next_beta);
    beta = next_beta;
    if cost > new_cost
        cost = new_cost;
    else
        break;
    end
    if itera > 20
        iterList = [iterList;itera];
        costList = [costList;cost];
    end
end

disp("Number of iteration = "+itera+" alpha = " + a);
plot(iterList,costList); %% plot cost as function over iterations
end

%% cost function
function cost = calculateC(X,y,B)
cost = ((-1)/size(X,1))*((y.')*log(a2.sigmoid(X*B))+((1-y).')*log(1-a2.sigmoid(X*B)));
end

%% function error calculation
function error = calculateE(X,B)
n = length(X(:,1:9));
XX = [ones(n,1),normalize(X(:,1:9))];
pre = a2.sigmoid(XX*B);
p = round(pre);
error = 0;
for i=1:length(p)
    if X(i,10) ~= p(i)
        error = error+1;
    end
end
end
