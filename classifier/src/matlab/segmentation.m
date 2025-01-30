%
% This file is used to wrap runSpringerSegmentationAlgorithm.m for use in python
% through the matlab engine
% Author : Milan Marocchi

function assigned_states = segmentation(signal, Fs)
    load('Springer_B_matrix.mat');
    load('Springer_pi_vector.mat');
    load('Springer_total_obs_distribution.mat');

    assigned_states = runSpringerSegmentationAlgorithm(double(signal), double(Fs), Springer_B_matrix, Springer_pi_vector, Springer_total_obs_distribution, false);
end