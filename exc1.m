a2.clear()
load('data_build_stories.mat')

%% ploting data
X = data_build_stories(:,1);
y = data_build_stories(:,2);
scatter(X,y,'.');

%% calculate beta
B = a2.calcB(X,y);

%% calculate J(B) by normal equation and gradiant descent
Normal_cost = J(X,y,B);
%Gradiant_cost = GradiantDNN(X,y,0.0000001,5000000);
%Gradiantf_cost = GradiantDWN(X,y,0.0000001,5000000);

%% calcualte and print predicated value
calcIteration(X,y,0.0000001,Normal_cost)
calcIterationWN(X,y,0.0000001,Normal_cost)

%% J(B) calculation
function mse = J(X,y,B)
sum = 0;
for i=1:length(X)
    sum = sum+((B(1,1)+B(2,1)*X(i,1))-y(i,1))^2;
end
mse = sum/length(X);

end

%% gradiant descent without normalization
function nNB = GradiantDNN(X,y,a,N)
B = [0;0];
n = length(X);
X2 = [ones(n,1),X];
cost = J(X,y,B);
i=0;
while N>i
    i=i+1;
    next_B = B-(a*(X2.')*((X2*B)-y));
    new_cost = J(X,y,next_B);
    if new_cost>cost
        break;
    else
      B = next_B; 
      cost = new_cost;
    end 
end
nNB = J(X,y,B);
end

%% Gradiant Descent with Normalization
function NB = GradiantDWN(X,y,a,N)
B = [0;0];
n = length(X);
cost = J(X,y,B);
M = mean(X);
S = std(X);
XX = [ones(n,1),ones(n,1)];
for i=1:length(X)
    XX(i,2) = (X(i)-M)/S;
end

i=0;
while N>i
    i=i+1;
    next_B = B-(a*(XX.')*((XX*B)-y));
    new_cost = J(X,y,next_B);
    if new_cost>cost
        break;
    else
      B = next_B; 
      cost = new_cost;
    end 
end
NB = J(X,y,B);
end

%% calc best iteration number
function iter = calcIteration(X,y,a,NC)
B = [0;0];
n = length(X);
X2 = [ones(n,1),X];
cost = J(X,y,B);
i=0;
while abs(NC-cost)> (NC*0.01)
    i=i+1;
    next_B = B-(a*(X2.')*((X2*B)-y));
    new_cost = J(X,y,next_B);
    if new_cost>cost
        break;
    else
      B = next_B; 
      cost = new_cost;
    end 
end

p = predict(B(1,1),B(2,1),900);
disp("(no normalization)Prediction of 900 is " + p);
iter = i;
end

%% calc best iteration number with normalization
function itera = calcIterationWN(X,y,a,NC)
B = [0;0];
n = length(X);
M = mean(X);
S = std(X);
cost = J(X,y,B);
XX = [ones(n,1),ones(n,1)];
for i=1:length(X)
    XX(i,2) = (X(i)-M)/S;
end

x=0;
while abs(NC-cost) > (NC*0.01)
    x=x+1;
    next_B = B-(a*(XX.')*((XX*B)-y));
    new_cost = J(X,y,next_B);
    if new_cost > cost
        break;
    else
      B = next_B; 
      cost = new_cost;
    end 
end

p = predict(B(1,1),B(2,1),((900-M)/S));
disp("Prediction of 900 is " + p);
itera = x;
end

%% predict
function p = predict(a,b,c)
p = a+(b*c);
end
