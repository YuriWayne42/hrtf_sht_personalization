% write fn for HRTF recreation
% suppose we have C for 128 positions
function recreated_HRTF = HRTF_SHT_recreat(C_matrix, azi, elev)
    %C_matrix is (L+1)^2 * 128
    %azi & elev is in degree
    SH_order = sqrt(size(C_matrix,1)) - 1;
    TH = azi/180*pi;
    PHI = elev/180*pi;
    
    SH_Vec = SHCreateYVec(SH_order, TH, pi/2 - PHI);
    Y = SH_Vec(:)';
    
    recreated_HRTF =  Y * C_matrix;

end