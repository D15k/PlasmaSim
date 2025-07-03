% Sizes to test
sizes = [2^7, 2^8, 2^9, 2^10, 2^11, 2^12, 2^13, 2^14, 2^15];
seed = 42;

fprintf('MATLAB PERFORMANCE RESULTS\n');
fprintf('----------------------------------------\n');
fprintf('Matrix Size\tMATLAB Time (s)\n');
fprintf('----------------------------------------\n');

for i = 1:length(sizes)
    N = sizes(i);

    % Set fixed seed
    rng(seed);

    % Generate same matrices for all runs
    A = rand(N, N);
    B = rand(N, N);

    % Warm-up
    C = A * B;

    % Measure time
    t = timeit(@() A * B);

    % Print result
    fprintf('%dx%d\t\t%.6f\n', N, N, t);
end
