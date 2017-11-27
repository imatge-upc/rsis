
%root_path = '/Users/miriambellverbueno/';
root_path = '/home/mbellver/'; 
gt_path = strcat(root_path, 'minotaure_scratch/LeafsDataset/A1/');
results_path = strcat(root_path, 'minotaure_data2/riass/models/34/34_results/A1/');
results_files = '*_label.png';

list_results = dir([results_path, results_files]);

% total_BestDice = 0;
% total_FGBGDice = 0;

total_AbsDiffFGLabels = 0;
total_SBD = 0;

total_files = length(list_results);

for i=1:total_files
    
    gt = imread([gt_path, list_results(i).name]);
    result = imread([results_path, list_results(i).name]);

    total_AbsDiffFGLabels = total_AbsDiffFGLabels + AbsDiffFGLabels(result, gt);
    total_SBD = total_SBD + SymmetricBestDice(result, gt);
    
end

avg_AbsDiffFGLabels = total_AbsDiffFGLabels/total_files;
avg_SBD = total_SBD/total_files;

fprintf('The average AbsDiffFGLabels is %.2f \n', avg_AbsDiffFGLabels);
fprintf('The average SBD is %.2f \n', avg_SBD);
