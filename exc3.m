a2.clear()
load('housing_price_index.mat')

%% ploting
X = housing_price_index(:,1);
y = housing_price_index(:,2);

for i=1:9
subplot(3,3,i)
title(i);
hold on
plot(X,y,'.');
polyAlgorithm(X,y,i);
end

%% best fitting and prediction
disp("I think polynomial 6 would be the best fitting one.");
disp("because it seems to neither over fit or under fit the training set");
disp("but i can't be sure since there ain't no test data to check the test MSE");

disp("prediction of his house seems fitting on the line draw on the diagram");
disp("but it is not realistic since if it is then Jonas would lost 0,8 million");


%% polynomial algorithm function
function poly = polyAlgorithm(X,y,d)

%n = length(X);
%XX = ones(n,1);
%for i=1:d
%    XX = [XX,X.^i];
%end

%beta = ((XX.'*XX)^(-1))*(XX.')*y;
%for i=1:d
%    yy= yy+(beta(i+1,1)*XX(:,i+1));
%end

[beta1,~,mu]= polyfit(X,y,d);
yyy = polyval(beta1,X,[],mu);
plot(X,yyy);
if d==6
    [beta1,~,mu]= polyfit(X,y,6);
    Z=[42,43,44,45,46,47,48,49,50];
    yyy1 = polyval(beta1,Z,[],mu);
    plot(Z,yyy1)
    yyy = polyval(beta1,48,[],mu);
    plot(48,yyy,'ro');
end

poly = 0;
end
