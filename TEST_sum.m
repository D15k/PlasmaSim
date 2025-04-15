 
Nx = 2^8;                    
Nv = 2^9;
Ns = 2;      

k      = 0.5;
alpha  = 0.5;
Ue     = -2; 

Lx     = 2 * pi / k;                   
Lv     = 8;      
                         
x = [0:Nx-1] * Lx / Nx; 
v = linspace(-Lv,Lv,Nv);

[X,V] = meshgrid(x, v);

pert = @(x) alpha * cos(k * x);
fini = @(x, v) (1+pert(x)) ./ (sqrt(2 * pi)) .* (exp(-(v - Ue).^2 / 2));

%% Matlab
fs(:, :) = fini(X, V);
rho_matlab = sum(fs(:,:))