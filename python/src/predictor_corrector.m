function [fs,params] = predictor_corrector(params,fs)

    iT = params.it+1;

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

    % save electric field;
    params.Efield = Efield;
    params.Efield_list(:,iT) = Efield;

    
end