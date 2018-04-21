a2.clear()
load('data_build_stories.mat')

X = data_build_stories(:,1);
y = data_build_stories(:,2);

a= 0.01;
n = length(X);
beta = calcIterationWN(X,y,a);
XX = [ones(n,1),a2.normalize(X)];
pre = ones(length(X),1);
y1 = ones();
for i=1:length(X)
    pre(i,1) = XX(i,:)*beta;
    y1(i,1) = y(i,1)- pre(i,1);
end

M = mean(y1);
subplot(2,1,1)
sortR = sortrows([X,y1],1);
plot(sortR(:,1),sortR(:,2));
grid on
subplot(2,1,2)
sortR = sortrows([y,y1],1);
plot(sortR(:,1),sortR(:,2));
grid on

corrlation = corr(X,y);

model = fitlm(X,y);
SE = model.Coefficients.SE;
B = model.Coefficients.Estimate;
int1= [B(1)-SE(1),B(1)+SE(1)];
int2= [B(2)-SE(2),B(2)+SE(2)];

%% mse
function mse = J(X,y,B)
sum = 0;
for i=1:length(X)
    sum = sum+((B(1,1)+B(2,1)*X(i,1))-y(i,1))^2;
end
mse = sum/length(X);

end

%% gradiant descent
function B = calcIterationWN(X,y,a)
n = length(X);
XX = a2.normalize(X);
B = a2.calcB(XX,y);
XX = [ones(n,1),XX];
NC = a2.J(XX,y,B);
B= zeros(size(XX,2),1);
cost = a2.J(XX,y,B);

x=0;
while abs(NC-cost) > (NC*0.01)
    x=x+1;
    next_B = B-(a*(XX.')*((XX*B)-y));
    new_cost = a2.J(XX,y,next_B);
    B = next_B;
    cost = new_cost;
end
end
