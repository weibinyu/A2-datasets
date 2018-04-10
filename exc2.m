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