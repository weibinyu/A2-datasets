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

%% cost function
function cost = calculateC(X,y,B)
cost = ((-1)/size(X,1))*((y.')*log(a2.sigmoid(X*B))+((1-y).')*log(1-a2.sigmoid(X*B)));
end