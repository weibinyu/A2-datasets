a2.clear()
load('GPUbenchmark.csv')

%% ploting all data
X = GPUbenchmark(:,1:6);
y = GPUbenchmark(:,7);
for i=1:6
    subplot(2,3,i)
    plot(GPUbenchmark(:,i),y,'.')
end

%% calculate B
B = a2.calcB(X,y);

%% calculate prediction
X1=[1;2432; 1607; 1683; 8; 8; 256];
pred = predict(X1,B);
% the error is because of the linje regressional nature, because the
% data is not only integer so there could be errors

%% cost
Normal_cost = J(X,y,B);
[p,GD] = calcIterationWN(X,y,0.01);



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
for i=2:size(X,1)
   pre = pre+(B(i,1)*X(i,1));
end

end

%% gradiant descent
function [p,itera] = calcIterationWN(X,y,a)
n = length(X);
M = mean(X);
S = std(X);
XX = a2.normalize(X);
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
