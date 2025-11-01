% BOOTSRAP_LOGISTIC_MEAN_AND_CI.m

% Bootstrap CIs for the mean under a Logistic(mu,s) population.
% Compares Normal (Fisher-info) CI vs Percentile Bootstrap CI.
clear; clc; close all; 


rng('default');

% --- Population (for simulation) ---
mu = 0; s = 1;                    % Logistic location & scale
n  = 80;                          % sample size
X  = mu + s * log(rand(n,1)./(1-rand(n,1)));  % logistic via inverse-CDF

% --- Point estimate and asymptotic (normal) CI using Var = (pi^2 s^2)/3 ---
xbar = mean(X);
s_hat = sqrt(var(X) * 3 / pi^2);  % plug-in for logistic scale
z = 1.96;
ci_norm = [xbar - z*sqrt((pi^2*s_hat^2)/(3*n)),  xbar + z*sqrt((pi^2*s_hat^2)/(3*n))];

% --- Bootstrap percentile CI ---
B = 2000;
boot_means = zeros(B,1);
for b = 1:B
    Xb = X(randi(n,n,1));
    boot_means(b) = mean(Xb);
end
ci_pct = quantile(boot_means, [0.025 0.975]);

% --- Display ---
fprintf('Sample mean      = %.4f\n', xbar);
fprintf('Normal (Wald) CI = [%.4f, %.4f]\n', ci_norm(1), ci_norm(2));
fprintf('Percentile CI    = [%.4f, %.4f]\n', ci_pct(1), ci_pct(2));

% --- Plot ---
figure('Color','w'); hold on; box on; grid on;
histogram(boot_means, 'Normalization','pdf', 'EdgeColor','none');
xline(xbar, 'k-', 'LineWidth',1.5, 'DisplayName','x̄');
xline(ci_pct(1), 'r--', 'LineWidth',1.5, 'DisplayName','Bootstrap 2.5%');
xline(ci_pct(2), 'r--', 'LineWidth',1.5, 'DisplayName','Bootstrap 97.5%');
xline(ci_norm(1),'b-.','LineWidth',1.5, 'DisplayName','Normal 2.5%');
xline(ci_norm(2),'b-.','LineWidth',1.5, 'DisplayName','Normal 97.5%');
title('Bootstrap distribution of the sample mean (Logistic population)');
legend('Location','best');
xlabel('Mean'); ylabel('Density');
