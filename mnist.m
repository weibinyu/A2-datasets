%% Assuming that the training data is loaded
% images is data matrix (60k x 784)
% labels is the labels (60k x 1)

%% Extract a smaller subset to start working with 
n = 1000; % smaller subset size
r = randperm(size(images,1));
X = images(r(1:n),:);
y = images(r(1:n));

%% Displaying images
% The following command displays a random image from the mnist data loaded
% into data matrix X
img_size = 28;
sel = randi(size(X,1));
figure(1)
imagesc(reshape(X(sel,:),img_size,img_size),[0,1])

% Same idea but 4 images
n = 4;
sel = randi(size(X,1),1,n);
figure(2)
imagesc(reshape(X(sel,:)', img_size, n*img_size),[0,1])


%% Classification
% The essential part is that we need to construct 10 different logistic
% classifiers one for each label [0-9].

for i=1:10
    % train one classifier per iteration 
end

%% Testing
% beta is here a (784 x 10) matrix containing all ten models. This command
% returns the class for each instance in the training data for which the
% estimated probability that an instance is in that class is highest. Hence
% from this you can compute the training error, i.e. p is an (60k x 1)
% containing the training predictions.
[~,p] = max(sigmoid(X*beta),[],2);

