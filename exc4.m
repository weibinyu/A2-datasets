a2.clear()
load('football.mat')

%% plot and test sigmoid

%plotFootball(football_X, football_y)
s = a2.sigmoid([0 1;2 3]);
i = 0;
while true
    if (a2.sigmoid(i)-1) == 0
        disp("Matlab interprets sigmoid("+i+")-1 as zero");
        break
    end
    i = i+1;
end


%% calculation cost when B = 0
beta = [0;0;0];
aX = [ones(size(football_X,1),1),football_X];
cost = calculateC(aX,football_y,beta);
[beta,itera] = findB(football_X,football_y,0.01);

disp("By observing the diagram with decision boundary. I see that training error");
disp("of the model is 7 and training accuracy is 98.11 percent");

%% cost function
function cost = calculateC(X,y,B)
cost = ((-1)/size(X,1))*((y.')*log(a2.sigmoid(X*B))+((1-y).')*log(1-a2.sigmoid(X*B)));
end

%% function finding B
function [beta,itera] = findB(X,y,a)
n = length(X);
M = mean(X);
S = std(X);
XX = [ones(n,1),ones(n,1)];
for i=1:size(X,1)
    XX(i,1) = (X(i,1)-M(1))/S(1);
    XX(i,2) = (X(i,2)-M(2))/S(2);
end
XX = [ones(n,1),XX];
beta = [0;0;0];
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

disp("Number of iteration = "+itera+" alpha = "+a);
plot(iterList,costList); %% plot cost as function over iterations

%plotFootball(X,y,beta,M,S);
beta = beta;
end