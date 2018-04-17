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

%% feature mapping
mapped = mapFeature(X1,X2,2);
B=[0;0;0;0;0;0;];
[beta,itera] = GD(mapped,y,B,0.01);
plot2dContour(beta,mapped,y,2);

%% gradiant descent
function [beta,itera] = GD(X,y,B,a)
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