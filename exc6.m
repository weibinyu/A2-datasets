a2.clear()
load('microchiptests.csv')
X1 = microchiptests(:,1);
X2 = microchiptests(:,2);
y = microchiptests(:,3);

%% plot
plot(X1,y,'ro')
hold on
plot(X2,y,'bx')
hold off
ylim([-1,2])

%% plot using gradient descent

d = 1;
B = initB(d);
mapped = mapFeature(X1,X2,d);

%[beta,itera,cost] = GD(mapped,y,B,0.01);
%plot2dContour(beta,mapped,y,d);


%% plot using fminunc
for i=1:9
d = i;
B = initB(d);
mapped = mapFeature(X1,X2,d);
fun = @(B)((-1)/size(mapped,1))*((y.')*log(a2.sigmoid(mapped*B))+((1-y).')*log(1-a2.sigmoid(mapped*B)));
%options = optimset('GradObj', 'on', 'MaxIter', 1000,'Display','on');
[theta, final_cost] = fminunc(fun, B);
subplot(3,3,i)
plot2dContour(theta,mapped,y,d)
title("degree is "+ i)
end

%% gradiant descent
function [beta,itera,cost] = GD(X,y,B,a)
cost = calculateC(X,y,B);
itera = 0;
while true
    itera = itera+1;
    next_B = B-(a*(X.')*(a2.sigmoid(X*B)-y));
    new_cost = calculateC(X,y,next_B);
    B = next_B;
    if cost > new_cost
        cost = new_cost;
    else
        break;
    end
end
beta = B;
disp("When alpha is "+ a + "and iterations is " + itera + " and B is");
disp(B);

end

%% calculate cost
function cost = calculateC(X,y,B)
cost = ((-1)/size(X,1))*((y.')*log(a2.sigmoid(X*B))+((1-y).')*log(1-a2.sigmoid(X*B)));
end

%% feature mapping function
function out = mapFeature(X1, X2, D)
out = ones(size(X1(:,1)));
for i = 1:D
    for j = 0:i
        out(:, end+1) = (X1.^(i-j)).*(X2.^j);
    end
end
end

%% plot decision boundary
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
