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
beta = [0;0];
cost = calculateC(football_X,football_y,beta);

<<<<<<< HEAD
%% cost function
function cost = calculateC(X,y,B)
cost = ((-1)/size(X,1))*((y.')*log(a2.sigmoid(X*B))+((1-y).')*log(1-a2.sigmoid(X*B)));
=======
[beta,itera] = findB(football_X,football_y,0.0001)


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
%%XX = [ones(n,1),XX];
beta = [0;0];
itera = 0;
while true
    itera = itera+1;
    cost =((-1)/size(XX,1))*((y.')*log(a2.sigmoid(XX*beta))+((1-y).')*log(1-a2.sigmoid(XX*beta)));
    next_beta = beta-(a*(XX.')*(a2.sigmoid(XX*beta)-y));
    new_cost = ((-1)/size(XX,1))*((y.')*log(a2.sigmoid(XX*next_beta))+((1-y).')*log(1-a2.sigmoid(XX*next_beta)));
    beta = next_beta;
    if cost > new_cost
        cost = new_cost;
    else
        break;
    end
end

beta = beta;
>>>>>>> origin/master
end