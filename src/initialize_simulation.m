function [params, fs] = initialize_simulation(params)
% Initialize grids and distribution functions
for s = 1:params.Ns
    params.Lv = params.Lv_s(s);
    params.x = [0:params.Nx-1] * params.Lx / params.Nx; % Spatial grid
    params.v = [0:params.Nv-1] * 2 * params.Lv / params.Nv - params.Lv; % Velocity grid
    params.dx = params.x(2) - params.x(1); % Spatial grid spacing
    params.dv = params.v(2) - params.v(1); % Velocity grid spacing
    % Create meshgrid for spatial and velocity coordinates
    [params.Xs{s}(:, :, 1), params.Xs{s}(:, :, 2)] = meshgrid(params.x, params.v);

    % Store grid information
    grid.Lv = params.Lv;
    grid.X = params.Xs{s}(:, :, 1);
    grid.V = params.Xs{s}(:, :, 2);
    grid.Mask = 1;
    grid.x = params.x;
    grid.dv = params.dv;
    grid.dx = params.dx;
    grid.Lx = params.Lx;
    grid.size = size(params.Xs{s}(:, :, 1));
    grid.dom = [0, -params.Lv, params.Lx, params.Lv];
    grid.method = "spline";
    params.grids(s) = grid;
end

% Initialize distribution functions

for s = 1:params.Ns
    fini = params.fini{s};
    fs(:, :, s) = fini(params.grids(s).X,params.grids(s).V);
end

% Directories
params.data_dir= "data/" + params.mycase+"_"+params.method+"/";
if ~exist(params.data_dir, 'dir')
    mkdir(params.data_dir);
end

end