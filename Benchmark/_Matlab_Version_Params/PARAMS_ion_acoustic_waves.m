params = struct();
params.mycase = "ion_accoustic_waves_weak";          % "two_stream"
params.Nx = 2^9;                            % Number of spatial grid points
params.Nv = 2^9;                            % Number of velocity grid points
params.Ns = 2;                              % Number of species (electrons and ions)
params.method="NuFI";
params.species_name = ["electrons","ions"]; % name of the different species
params.Mr = 1000;                           % Mass ratio for ions
params.Mass = [1, params.Mr];               % Mass of species
params.charge = [-1, 1];                    % Charge of species
params.Nt_max = 5000;                       % Maximum number of time steps
params.dt = 10;                            % Time step size
params.Nsubs = 8;
params.dt_save = params.dt;                       % Save after dt_save time
params.Tend = 5000;                          % End time of simulation

% Initial condition parameters
params.k = 0.5;                             % Wave number
params.alpha = 0.01;                         % Perturbation amplitude
params.pert = @(x) params.alpha * cos(params.k * x); % Perturbation function

% Electrons
params.Ue = -2;                   % Electron drift velocity
params.fe0 = @(x, v) (1+params.pert(x)) ./ (sqrt(2 * pi)) .* (exp(-(v - params.Ue).^2 / 2)); % Electron distribution
params.fi0 = @(x, v) sqrt(params.Mr / (2 * pi)) .* (exp(-params.Mr * (v).^2 / 2)); % Ion distribution
params.fini = {params.fe0,params.fi0};
% Grid parameters
params.Lx = 2 * pi / params.k;    % Spatial domain length
params.Lv_s = [8, 0.1 * pi]; % Velocity domain lengths for each species
