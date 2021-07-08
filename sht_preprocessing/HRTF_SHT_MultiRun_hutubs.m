% HUTUBS database contain 440 points, convering all spatial directions
% no need for conformal mapping
% still using dB scale

%generate the least square fit for the hutubs HRTF and recreate it
 

% for original ITA database, 2304 points = 72*32, 
%the azimuth angle is 5 degree: [0,360]; the elevation angle is 5.04 degree: [90, -66]
% so, will add -75, -80, -85 elevations


%f = [f(phi1,theta1),f(phi2,theta2)...]'
%c = [c00, c1-1, c10, c11...]'
%Y = [y00, y1-1, y10, y11...]  = SHbase_P
%ynm = [Ynm(phi1,theta1),...]'

% C_matrix is conbination of c, for 128 freqs
tstart = tic;
% load('ITA_matrix.mat','hrtf_ITA_all');
 

path = 'HRIRs\';
SH_order = 7;
hrtf_SHT_dBmat = zeros(96, 128, (SH_order+1)^2, 2);

% hrtf_dir = dir([path,'*measured.sofa'])

hrtf_dir = dir([path,'*simulated.sofa']);
% function [C_matrix_l, C_matrix_r] = HRTF_SHT_MultiRun(SH_order)
% load('hzmHRTF.mat');
% load('input_hrtf_mag.mat');
% SH_order = 4;
% C_matrix_l = zeros( (SH_order+1)^2, 128);
% C_matrix_r = zeros( (SH_order+1)^2, 128);

% theta = linspace(0, pi, 181); 
% phi = linspace(0, 2*pi, 361); 
% [tt,pp]=meshgrid(theta,phi);

% N = (SH_order+1)^2 *2; %try N = 1+ l*(l+1)
% N = (SH_order+1)^2*2; 
% now we need ALL!

str = 'running';
hwait=waitbar(0,str);


for ind = 1:length(hrtf_dir)
    waitbar( ind/length(hrtf_dir) , hwait,str);
    
    hrtfData = SOFAload(strcat(path,'\', hrtf_dir(ind).name)  );
    
N = size(hrtfData.SourcePosition, 1);


epsilon = 1e-6;               %
% phi_sample = (sqrt(5)-1)/2;


hrtf_freq_l = fft( squeeze(hrtfData.Data.IR(:,1,:) )' );
hrtf_freq_r = fft( squeeze(hrtfData.Data.IR(:,2,:) )' );

hrtf_freq_l = hrtf_freq_l';
hrtf_freq_r = hrtf_freq_r';


 
input_locations_sph = deg2rad(hrtfData.SourcePosition(:,1:2));

TH = input_locations_sph(:,1);
PHI = input_locations_sph(:,2);
 

SHbase_P = 0;
SHbase_P(1:N,1:(SH_order+1)^2) = 0; 
SHbase_P(1:N,1) = 1;   %0 order is 1 anyway


for i = 1:N
    SH_Vec = SHCreateYVec(SH_order, TH(i), pi/2 - PHI(i));
    SHbase_P (i, :) = SH_Vec'; 
%     SHbase_P(i ,n^2+k+n+1) = PP(Sample_coor(i,2), Sample_coor(i,1)); 
end

SHbase_P = roundn(SHbase_P, -5);

for i_freq = 1:128
    % frequency_bin = 16;
    % FF = roundn(FF, -5);
 
    f1 = abs(hrtf_freq_l(:, i_freq));
    f2 = abs(hrtf_freq_r(:, i_freq));
    
    % change to dB scale in this version 
    f1 = mag2db(f1 ./ 20e-6 );
    f2 = mag2db(f2 ./ 20e-6 );
    
    
%     C1 = [];
%     C2 = [];
    % C = (SHbase_P.'*SHbase_P)\(SHbase_P.'*f);   %calc the weighting coeff
    
    %testing on the Tikhonov Regulization
    C1 = (SHbase_P.'*SHbase_P + epsilon* eye((SH_order+1)^2))\(SHbase_P.'*f1);
    C2 = (SHbase_P.'*SHbase_P + epsilon* eye((SH_order+1)^2))\(SHbase_P.'*f2);
    
%     C_matrix_l(:,i_freq) = C1;
%     C_matrix_r(:,i_freq) = C2;
    
    
%     hrtf_SHT_mat = zeros(48, 128, (SH_order+1)^2, 2);

    hrtf_SHT_dBmat(ind, i_freq, :, 1) = C1;
    hrtf_SHT_dBmat(ind, i_freq, :, 2) = C2;
    


end
end 


telapsed = toc(tstart);

 