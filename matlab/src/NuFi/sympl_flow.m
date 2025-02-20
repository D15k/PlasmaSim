%

function [X,V] = sympl_flow(n, dt, X,V, ZEfield, grid)
mint="spline";
if n == 1
     return;
end

size_grid = size(V);

% Omit the following line for Psi_tilda:
V = V + (dt / 2) * reshape(interp1(grid.x,ZEfield(:,n),reshape(X,[],1),mint),size_grid);

while n > 2
    n = n - 1;
    X = X - dt * V;  % Inverse signs; going backwards in time
    X = mod(X,grid.Lx-grid.dx); % assuming periodic grid
    V = V + dt *reshape(interp1(grid.x,ZEfield(:,n),reshape(X,[],1),mint),size_grid);
end

X = X - dt * V;
X = mod(X,grid.Lx-grid.dx);
V = V + (dt / 2) *reshape(interp1(grid.x,ZEfield(:,1),reshape(X,[],1),mint),size_grid);

end