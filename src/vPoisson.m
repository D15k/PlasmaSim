function [Efield] = vPoisson(fs, grids, charge)
    
% compute charge density
rho = 0;
Ns = length(grids);
for s = 1:Ns
    rho = rho + charge(s)*sum(fs(:,:,s))*grids(s).dv;
end

Nx = grids(1).size(1);
kxn =  (mod((1:Nx)-ceil(Nx/2+1),Nx)- floor(Nx/2)) ;
Lx = grids(1).Lx;
rx = 2*pi*1i./Lx;
kx = kxn.*rx;

% laplacian is division -|k|^2
K2 = kx'.^2;
K2 ( abs(K2) < 1.0e-11 ) = 1;
% to avoid a division by zero, we set the zeroth wavenumber to one.
% this leaves it's respective Fourier coefficient unaltered, so the
% zero mode of Sk is conserved.dphi_dx_h = 1i*phi_fft.*kx(1,:); This way, Sk's zero mode implicitly
% defined the zero mode of the result
% Note that the zero mode is NOT uniquely defined: in a periodic
% setting, the solution of Laplace's (or Poisson's) equation is only
% defined up to a constant! You can freely overwrite the zero mode,
% therefore.
b = fft(rho); 
b(1) = 0;
phi_fft = - b' ./ K2; % solves second equation of vlassov poisson
dphi_dx_h = phi_fft.*kx';
    Efield = -reshape(ifft(dphi_dx_h, "symmetric"), 1, []);
    
end