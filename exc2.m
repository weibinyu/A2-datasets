a2.clear()
load('GPUbenchmark.csv')

%% ploting all data
X = GPUbenchmark;
X(:,7)=[];
y = GPUbenchmark(:,7);
for i=1:6
    subplot(2,3,i)
    plot(GPUbenchmark(:,i),y,'.')
end

%% calculate B
B = a2.calcB(X,y);

%% calculate prediction
X1=[2432; 1607; 1683; 8; 8; 256];
pred = predict(X1,B);
% the error is because of the linje regressional nature, because the
% data is not only integer so there could be errors

%% cost
Normal_cost = J(X,y,B);
[p,GD] = calcIterationWN(X,y,0.001);



%% normal equation.
function mse = J(X,y,B)
sum = 0;
for i=1:length(X)
    sum = sum+(predict(X(i,:)',B)-y(i))^2;
end
mse = sum/length(X);

end

%% prediction
function pre = predict(X,B)
pre = B(1,1);
for i=1:6
   pre = pre+(B(i+1,1)*X(i,1));
end
end

%% gradiant descent
function [p,itera] = calcIterationWN(X,y,a)
n = length(X);
M = mean(X);
S = std(X);
XX = [ones(n,1),ones(n,1),ones(n,1),ones(n,1),ones(n,1),ones(n,1)];
for i=1:size(X,1)
    XX(i,1) = (X(i,1)-M(1))/S(1);
    XX(i,2) = (X(i,2)-M(2))/S(2);
    XX(i,3) = (X(i,3)-M(3))/S(3);
    XX(i,4) = (X(i,4)-M(4))/S(4);
    XX(i,5) = (X(i,5)-M(5))/S(5);
    XX(i,6) = (X(i,6)-M(6))/S(6);    
end

B = a2.calcB(XX,y);
XX = [ones(n,1),XX];
NC = J(XX,y,B);
B=[0;0;0;0;0;0;0];
cost = J(XX,y,B);

x=0;
while abs(NC-cost) > (NC*0.01)
    x=x+1;
    next_B = B-(a*(XX.')*((XX*B)-y));
    new_cost = J(XX,y,next_B);
    B = next_B;
    cost = new_cost;
end

Z =[(2432-M(1))/S(1); (1607-M(2))/S(2); (1683-M(3))/S(3); (8-M(4))/S(4); (8-M(5))/S(5); (256-M(6))/S(6)];
p = predict(Z,B);
disp("Prediction by  GD is " + p);
itera = x;
end
