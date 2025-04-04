function [grid] = make_periodic_grid(Lx,Lv,Nx,Nv)

    x = [0:Nx-1] * Lx / Nx; % Spatial grid
    v = linspace(-Lv,Lv,Nv);

    dx = x(2) - x(1); % Spatial grid spacing
    dv = v(2) - v(1); % Velocity grid spacing

    % Create meshgrid for spatial and velocity coordinates
    [X,V] = meshgrid(x, v);


    grid.x  = x; grid.v  = v;
    grid.X  = X; grid.V  = V;
    grid.dx = dx;grid.dv = dv;
    grid.Lx = Lx;grid.Lv = Lv;
    grid.Nx = Nx;grid.Nv = Nv;

    grid.size = size(X);

    grid.dom = [0, -Lv, Lx-dx, Lv-dv];
    
end