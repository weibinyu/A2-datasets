a2.clear()
data = load("breast-cancer.mat");
data = data.breast_cancer;
data = data(randperm(size(data,1)),:); % Shuffle rows