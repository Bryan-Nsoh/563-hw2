function samples = slice1d(logpdf, x0, w, maxSteps, N, bounds)
% Simple univariate slice sampler with stepping-out and shrinkage.
%  logpdf  : handle to log-density (unnormalized)
%  x0      : starting point (within bounds)
%  w       : initial bracket width
%  maxSteps: max stepping-out steps
%  N       : number of samples to draw
%  bounds  : [L U] support (e.g., [-0.999, 0.999])
L = bounds(1); U = bounds(2);
samples = zeros(N,1);
x = min(max(x0,L),U);

for t = 1:N
    % vertical level
    logy = logpdf(x) + log(rand);
    % create interval [a,b] around x
    u = rand*w; a = x - u; b = x + (w-u);
    % step-out
    j = floor(rand*maxSteps);
    k = (maxSteps-1) - j;
    while j>0 && a>L && logpdf(a) > logy
        a = max(a - w, L); j = j - 1;
    end
    while k>0 && b<U && logpdf(b) > logy
        b = min(b + w, U); k = k - 1;
    end
    % shrinkage
    while true
        x_new = a + rand*(b-a);
        if logpdf(x_new) >= logy
            x = x_new; break;
        else
            if x_new < x, a = x_new; else, b = x_new; end
        end
    end
    samples(t) = x;
end
end
