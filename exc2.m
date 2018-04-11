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
GD = calcIterationWN(X,y,0.01);


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
function itera = calcIterationWN(X,y,a)
n = length(X);
M = mean(X);
S = std(X);
XX = [ones(n,1),ones(n,1),ones(n,1),ones(n,1),ones(n,1),ones(n,1)];
for i=1:length(X)
    XX(i,1) = (X(i,1)-M)/S;
    XX(i,2) = (X(i,2)-M)/S;
    XX(i,3) = (X(i,3)-M)/S;
    XX(i,4) = (X(i,4)-M)/S;
    XX(i,5) = (X(i,5)-M)/S;
    XX(i,6) = (X(i,6)-M)/S;    
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
    if new_cost < cost
        B = next_B;
        cost = new_cost;
    else
        break
    end
    
end

%p = predict(B(1,1),B(2,1),((900-M)/S));
%disp("Prediction of 900 is " + p);
itera = x;
end
