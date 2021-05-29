% this is a version for reference, GT

% HUTUBS database contain 440 points, convering all spatial directions
% no need for conformal mapping
% still using dB scale


%generate the least square fit for the HUTUBS HRTF and recreate it
% with all the points in the data

%f = [f(phi1,theta1),f(phi2,theta2)...]'
%c = [c00, c1-1, c10, c11...]'
%Y = [y00, y1-1, y10, y11...]  = SHbase_P
%ynm = [Ynm(phi1,theta1),...]'
[input_mat_name, pathname] = uigetfile('*.sofa','Pick an hrtf file in sofa');
hrtfData = SOFAload(input_mat_name);
 
% HUTUBS uses spherical source locations in beginning
 
input_locations_sph = deg2rad(hrtfData.SourcePosition(:,1:2));
TH = input_locations_sph(:,1);
PHI = input_locations_sph(:,2);

% check left hrtf just for a demo
input_hrtf = fft( squeeze(hrtfData.Data.IR(:,1,:) )' );
input_hrtf = input_hrtf';

% input_hrtf_mag = angle(input_hrtf); 
input_hrtf_mag = abs(input_hrtf); 

tstart = tic;
 
 
freq_ind = 42;
SH_order = 7;
 

N = length(TH);
freq_vec = linspace(0, 44100/2, 129);


%% select a single frequency hrtf 

% hrtfMag_singleF = abs(input_hrtf_mag(:,freq_ind));

hrtfMag_singleF = input_hrtf_mag(:,freq_ind);

hrtfMag_singleF = hrtfMag_singleF(:);


% hrtfMag_singleF = unwrap(hrtfMag_singleF);
% hrtfMag_singleF = mag2db(hrtfMag_singleF ./ 20e-6 ); %./ 20e-6
% hrtfMag_singleF = (hrtfMag_singleF.^(2/3.16) );

%% assign f and plot it at 231
f = hrtfMag_singleF;
f = f(:);

figure; 
subplot(231); 
plotSphFunctionTriangle_edited(f, [TH,PHI]);

cmap = getPyPlot_cMap('RdBu_r', 128); colormap(cmap)
 
% axis equal;
title(strcat( 'Original hrtf radial pattern, freq =', num2str( freq_vec(freq_ind)) )); %'FontSize',14, 'FontName', 'Arial'
 

%% compute SHT, and plot recreated mesh at 234
 
%testing on the Tikhonov Regulization
% C = (SHbase_P.'*SHbase_P + epsilon* eye((SH_order+1)^2))\(SHbase_P.'*f);
[C , f_recons] = SHT_core(f, [TH,PHI], SH_order);

subplot(234);
SHT_recons_mesh(C);
title(strcat('Recons shape, SH order = ',num2str(SH_order))); %'FontSize',14, 'FontName', 'Arial'

%% plot original v. recons at 232

subplot(232); plot(f); 
hold on; plot(f_recons);
error_rms = rms(f - f_recons);
legend('original','reconstructed','location','best')

title(strcat('sample number = ',num2str(N), ', RMS error =',num2str(error_rms)))

%% plot recons error at 235
subplot(235);
 
Recons_error = f - f_recons;

plotSphFunctionTriangle_edited(abs(Recons_error), [TH,PHI]);
hold on;
% draw a line pointing left 
x_line = [0,0];
y_line = [0,max(abs(Recons_error))*1.5];
z_line = [0,0];

line(x_line,y_line,z_line,'color','r','linewidth',2);

 
colorbar;
title('error distribution on space')

%% plot SHT coeff at 233
subplot(233); plot(C); title('weights of each SH coeff')

%% plot SHT coeff in pyramid form at 236
subplot(236);
SH_pyramidPlot(C);
% plot SH coeff in pyramid form
 
title('SHT coeff in pyramid grid')

cmap = getPyPlot_cMap('RdBu_r', 128);
colormap(cmap)

set(gcf,'position',[80 100 1200 600]);
% print(gcf,'-dpng',strcat('SHT_',num2str(SH_order),'_',num2str(N),'_conformalMap.png')) %'SHT_5_72.png'


telapsed = toc(tstart);
display(telapsed);

% title(strcat('Reconstructed shape, sample number = ',num2str(N),', SH order = ',num2str(SH_order),'error =',num2str(error_rms)),'FontSize',14, 'FontName', 'Arial')

