%
% 11/15/2016
% Libor Novak
%
% Test of ROI generation coverage
%

%% Settings
side = 200;
num_random_rois = 100;

%% Generation of multiple random rois
counts = zeros(side, side);

% Generate random rois
dim = floor(side/3);

for i = 1:num_random_rois
    x = ceil(rand() * (side-dim));
    y = ceil(rand() * (side-dim));
    
    counts(y:y+dim, x:x+dim) = counts(y:y+dim, x:x+dim) + 1;
end

counts = counts ./ max(max(counts));

imshow(counts);