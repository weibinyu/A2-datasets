a2.clear()
load('football.mat')

%% plot and test sigmoid
%plotFootball(football_X, football_y)
s = a2.sigmoid([0 1;2 3]);
i = 0;
while true
    if (a2.sigmoid(i)-1) == 0
        disp(i);
        break
    end
    i = i+1;
end
% Matlab interprets sigmoid(37)-1 as zero

%% calculation cost
beta = [0;0];
jb=((-1)/size(football_X,1))*((football_y.')*log(a2.sigmoid(football_X*beta))+((1-football_y).')*log(1-a2.sigmoid(football_X*beta)));

