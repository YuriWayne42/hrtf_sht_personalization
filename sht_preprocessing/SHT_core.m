%wrap SHT core function
function [C , f_recons] = SHT_core(f, dirs, SH_order)
% core function to process sht
% f - orginal data
% dirs - [azimuth1 elevation; ...; azimuthK elevation] angles in rads
%         for each evaluation point 


% f = Y*c
% so, c = Y^'-1' * f

%f = [f(phi1,theta1),f(phi2,theta2)...]'
%c = [c00, c1-1, c10, c11...]'
%Y = [y00, y1-1, y10, y11...]  = SHbase_P
%ynm = [Ynm(phi1,theta1),...]'
epsilon = 1e-6;               %



f = f(:);
TH = dirs(:,1);
PHI = dirs(:,2);


N = length(f);

SHbase_P = [];
SHbase_P(1:N,1:(SH_order+1)^2) = 0; 
SHbase_P(1:N,1) = 1;   %0 order is 1 anyway


for i = 1:N
    SH_Vec = SHCreateYVec(SH_order, TH(i), pi/2 - PHI(i));
    SHbase_P (i, :) = SH_Vec'; 
%     SHbase_P(i ,n^2+k+n+1) = PP(Sample_coor(i,2), Sample_coor(i,1)); 
end

% SHbase_P = roundn(SHbase_P, -5);


% add  Tikhonov Regulization
 C = (SHbase_P.'*SHbase_P + epsilon* eye((SH_order+1)^2))\(SHbase_P.'*f);

 f_recons = SHbase_P * C;

end