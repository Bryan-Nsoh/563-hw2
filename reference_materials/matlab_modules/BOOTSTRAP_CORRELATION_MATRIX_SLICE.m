%BOOTSTRAP_CORRELATION_MATRIX_SLICE.m

% Bootstrap the full dxd correlation matrix from a dataset,
% then slice-sample the marginal density of a chosen correlation r_{j,k}.

clear; clc; close all; 

% --- Data (You should replace with own real data) ---
% Example: generate correlated normal data with d=5

rng('default');

n = 200; d = 5;
Sigma = toeplitz(0.7.^(0:d-1));     % AR(1)-like correlation
X = mvnrnd(zeros(1,d), Sigma, n);

% --- Bootstrap of correlation matrix ---
B = 1500;
% Store upper-triangular (off-diagonal) elements: p = d(d-1)/2
p = d*(d-1)/2;
Rvec = zeros(B, p);
inds = find(triu(true(d),1));  % linear indices for upper triangle

for b = 1:B
    idx = randi(n, n, 1);      % resample rows
    Xb  = X(idx, :);
    Rb  = corr(Xb);            % dxd
    Rvec(b,:) = Rb(inds);      % vectorize unique correlations
end

% --- Choose a pair (j,k) to study with slice sampling ---
j = 1; k = 3;
linIdx = find(triu(true(d),1), 1, 'first'); % template
% Map (j,k) to position in Rvec columns
colmap = zeros(d); colmap(inds) = 1:p;
col = colmap(j,k);
r_boot = Rvec(:, col);

% --- Compare to Fisher-z interval ---
z = 0.5*log((1+r_boot)./(1-r_boot));          % transform bootstrap draws (for display)
z_mean = mean(z); z_se = 1/sqrt(n-3);
z_ci = [z_mean - 1.96*z_se, z_mean + 1.96*z_se];
ci_fisher = (exp(2*z_ci)-1)./(exp(2*z_ci)+1);

% --- Smooth a KDE for r in (-1,1), define unnormalized density handle ---
grid_r = linspace(-0.999,0.999,1000);
[kde_pdf, grid_r2] = ksdensity(r_boot, grid_r, 'Support',[-0.999,0.999]);
pdf_fun = @(r) interp1(grid_r2, kde_pdf, r, 'linear', 0);

% --- Univariate slice sampling over r in (-1,1) ---
Ns = 5000; burn = 1000;
r0 = median(r_boot);
logpdf = @(r) log(max(pdf_fun(r), realmin));     % safe log
% Step-out and shrinkage parameters
w = 0.05; maxSteps = 50;
samples = slice1d(logpdf, r0, w, maxSteps, Ns+burn, [-0.999, 0.999]);
r_slice = samples(burn+1:end);

% --- Intervals and plots ---
ci_boot = quantile(r_boot, [0.025 0.975]);
ci_slice = quantile(r_slice, [0.025 0.975]);

fprintf('Correlation r_{%d,%d}\n', j, k);
fprintf('Bootstrap percentile CI : [%.3f, %.3f]\n', ci_boot(1), ci_boot(2));
fprintf('Fisher-z CI (analytic)  : [%.3f, %.3f]\n', ci_fisher(1), ci_fisher(2));
fprintf('Slice-sampled CI        : [%.3f, %.3f]\n', ci_slice(1), ci_slice(2));

figure('Color','w'); 
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

% Left: bootstrap histogram + CIs
nexttile; hold on; box on; grid on;
histogram(r_boot, 'Normalization','pdf', 'EdgeColor','none');
xline(ci_boot(1),'r--','LineWidth',1.4,'DisplayName','Boot 2.5%');
xline(ci_boot(2),'r--','LineWidth',1.4,'DisplayName','Boot 97.5%');
xline(ci_fisher(1),'b-.','LineWidth',1.4,'DisplayName','Fisher-z 2.5%');
xline(ci_fisher(2),'b-.','LineWidth',1.4,'DisplayName','Fisher-z 97.5%');
xlabel(sprintf('r_{%d,%d}',j,k)); ylabel('Density');
title('Bootstrap distribution of r and analytic Fisher-z CI');
legend('Location','best');

% Right: slice-sampled marginal
nexttile; hold on; box on; grid on;
histogram(r_slice, 'Normalization','pdf', 'EdgeColor','none');
xline(ci_slice(1),'m--','LineWidth',1.4,'DisplayName','Slice 2.5%');
xline(ci_slice(2),'m--','LineWidth',1.4,'DisplayName','Slice 97.5%');
xlabel(sprintf('r_{%d,%d}',j,k)); ylabel('Density');
title('Slice-sampled marginal for r');
legend('Location','best');
