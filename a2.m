classdef a2 % Same name as .m file
    properties % Not in use
    end
    methods(Static)
        function clear()
            clear; % Clear Command Window
            close all; % Close all figure windows
            clc % Clear Workspace
        end
        
        function g = sigmoid(z)
        g = 1./(1+exp(-z));
        end
        
        function beta = calcB(X,y)%calculate beta using X and y
        n = length(X);
        X2 = [ones(n,1),X];
        beta = ((X2.'*X2)^(-1))*(X2.')*y;
        end
    end
end