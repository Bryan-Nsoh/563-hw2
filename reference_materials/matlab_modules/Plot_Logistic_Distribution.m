%% Plot_Logistic_Distribution.m
% Visualization of the Logistic distribution for various (mu, s)
% STAT THEORY & COMPUTATION — Project Exercise
% --------------------------------------------------------------
clear; clc; close all;

% x-grid for plotting
x = linspace(-8, 8, 400);

% Parameter sets
mu_vals = [0 0 0 -2 0 2];     % vary both location & scale
s_vals  = [0.5 1 2  1  1 1];  % first 3: vary scale, last 3: vary location
colors  = lines(length(mu_vals));

figure('Color','w'); hold on; box on; grid on;

for i = 1:length(mu_vals)
    mu = mu_vals(i); s = s_vals(i);
    fx = exp(-(x - mu)/s) ./ (s .* (1 + exp(-(x - mu)/s)).^2);
    plot(x, fx, 'LineWidth', 2, 'Color', colors(i,:), ...
         'DisplayName', sprintf('\\mu=%.1f, s=%.1f', mu, s));
end

% Overlay standard normal for comparison
fx_norm = normpdf(x,0,1);
plot(x, fx_norm, 'k--', 'LineWidth', 1.5, 'DisplayName','Normal(0,1)');

xlabel('x'); ylabel('Density f(x|\mu,s)');
title('Logistic Distribution for Different Location and Scale Parameters');
legend('Location','northeast');
set(gca,'FontSize',11);
