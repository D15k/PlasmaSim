function [fs, params] = NuFi(params,fs)
iT = params.it+1;
dt = params.dt;
for s = 1:params.Ns
    [X,V] = sympl_flow_Half(iT,dt,params.grids(s).X,params.grids(s).V,params.charge(s)/params.Mass(s)*params.Efield_list,params.grids(s));
    fini = params.fini{s};
    fs(:,:,s) = fini(X,V);
end
[Efield] =vPoisson(fs,params.grids,params.charge);
params.Efield = Efield;
params.Efield_list(:,iT) = Efield;
% for s = 1:params.Ns
%     [X,V] = sympl_flow(iT+1,dt,params.grids(s).X,params.grids(s).V,params.charge(s)/params.Mass(s)*params.Efield_list,params.grids(s));
%     fini = params.fini{s};
%     fs(:,:,s) = fini(X,V);
% end

end


function [X,V] = sympl_flow(n, dt, X,V, Efield, grid)
mint="spline";
if n == 1
     return;
end

% Omit the following line for Psi_tilda:
V = V + (dt / 2) * reshape(interp1(grid.x,Efield(:,n),reshape(X,[],1),mint),grid.size);

while n > 2
    n = n - 1;
    X = X - dt * V;  % Inverse signs; going backwards in time
    X = mod(X,grid.Lx-grid.dx); % assuming periodic grid
    V = V + dt *reshape(interp1(grid.x,Efield(:,n),reshape(X,[],1),mint),grid.size);
end

X = X - dt * V;
X = mod(X,grid.Lx-grid.dx);
V = V + (dt / 2) *reshape(interp1(grid.x,Efield(:,1),reshape(X,[],1),mint),grid.size);

end



function [X,V] = sympl_flow_Half(n, dt, X,V, Efield, grid)
mint="spline";
if n == 1
    return;
end

while n > 2
    n = n - 1;
    X = X - dt * V;  % Inverse signs; going backwards in time
    X = mod(X,grid.Lx-grid.dx); % assuming periodic grid
    V = V + dt *reshape(interp1(grid.x,Efield(:,n),reshape(X,[],1),mint),grid.size);
end

X = X - dt * V;
X = mod(X,grid.Lx-grid.dx);
V = V + (dt / 2) *reshape(interp1(grid.x,Efield(:,1),reshape(X,[],1),mint),grid.size);

end

