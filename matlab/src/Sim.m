function params = Sim(params)

% Initialize grids and distribution functions
[params, fs] = initialize_simulation(params);

% Plot initial conditions
params.Efield = vPoisson(fs,params.grids,params.charge);
plot_results(params, fs);
% Main loop over time
params.Efield_list(:,1) = params.Efield;
for iT = 1:params.Nt_max
    params.it = iT;
    % Perform a single time step
    [fs, params] = step(params, fs);

    % Measurements
    [params]=measure(params, fs);

    % Plot results at each time step
    plot_results(params, fs);

    % Check if simulation end time is reached
    time = params.dt * iT
    if time >= params.Tend
        break;
    end
end

end
%% Helper Functions

function [fs, params] = step(params, fs)


    if params.method == "predcorr"
        [fs,params] = predictor_corrector(params,fs);
    elseif params.method == "NuFi"
        [fs,params] = NuFi(params,fs);
    else
        display("error step")
    end


end






function plot_results(params, fs)
    % Plot results at each time step
    for s = 1:params.Ns
    subplot(params.Ns+1, 1, s);
    pcolor(params.Xs{s}(:, :, 1), params.Xs{s}(:, :, 2), fs(:, :, s)); shading flat;
    subtitle("$f_\mathrm{"+params.species_name(s)+"}$");
    colorbar();
    xlabel("$x$");
    ylabel("$v$");
    end
   
    subplot(params.Ns+1, 1, params.Ns+1);
    plot(params.grids(1).x, params.Efield); xlim([params.grids(1).x(1), params.grids(1).x(end)]);
    subtitle("$E$");
    colorbar();
    xlabel("$x$");

    pause(0.01); % Pause for visualization
end


