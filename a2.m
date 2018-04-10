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
    end
end