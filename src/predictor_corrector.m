function [fs,params] = predictor_corrector(params,fs)

    % Compute electric field
    [Efield] = vPoisson(fs, params.grids, params.charge);
    

    % Advect distribution functions for half time step
    for s = 1:params.Ns
        f12(:, :, s) = Advect(fs(:, :, s), params.charge(s) / params.Mass(s) * Efield, params.grids(s), params.dt / 2);
    end

    % Recompute electric field
    [Efield] = vPoisson(f12, params.grids, params.charge);

    % Advect distribution functions for full time step
    for s = 1:params.Ns
        fs(:, :, s) = Advect(fs(:, :, s), params.charge(s) / params.Mass(s) * Efield, params.grids(s), params.dt);
    end
    params.Efield = Efield;
end