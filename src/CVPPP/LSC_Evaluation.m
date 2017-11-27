%   Author: Hanno Scharr, Institute of Bio- and Geosciences II,
%   Forschungszentrum Jülich
%   Contact: h.scharr@fz-juelich.de
%   Date: 21.5.2014
%
% Copyright 2014, Forschungszentrum Jülich GmbH, 52425 Jülich, Germany
% 
%                          All Rights Reserved
% 
% All commercial use of this software, whether direct or indirect, is
% strictly prohibited including, without limitation, incorporation into in
% a commercial product, use in a commercial service, or production of other
% artifacts for commercial purposes.     
%
% Permission to use, copy, modify, and distribute this software and its
% documentation worldwide for research purposes solely is hereby granted 
% without fee, provided that the above copyright notice appears in all 
% copies and that both that copyright notice and this permission notice 
% appear in supporting documentation, and that the name of the author and 
% Forschungszentrum Jülich GmbH not be used in advertising or publicity 
% pertaining to distribution of the software without specific, written 
% prior permission of the author available under above contact.
% The author preserves the rights to request the deletion or cancel of 
% non-authorized advertising and /or publicity activities.
%
% For intentions of commercial use please contact 
%
% Forschungszentrum Jülich GmbH
% To the attention of Hans-Werner Klein
% Department Technology-Transfer
% Wilhelm-Johnen-Straße
% 52428 Jülich
% Germany
%
%
% THE AUTHOR AND FORSCHUNGSZENTRUM JÜLICH GmbH DISCLAIM ALL WARRANTIES WITH 
% REGARD TO THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF 
% MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE.  IN NO EVENT 
% SHALL THE AUTHOR OR FORSCHUNGSZENTRUM JÜLICH GmbH BE LIABLE FOR ANY SPECIAL,  
% INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING 
% FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, 
% NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH
% THE USE OR PERFORMANCE OF THIS SOFTWARE.  
%
% In case of arising software bugs neither the author nor Forschungszentrum 
% Jülich GmbH are obliged for bug fixes and other kinds of support.



% Evaluation of results provided by CVPPP Leaf Segmentation Contest
function LSC_Evaluation(inpath, gtpath)
% inpath: folder name, where subfolders for each participant can be found.
%         Subfolder names are used as unique identifier for each participant
% gtpath: folder name, where ground truth files can be found. subfolders
%         are assumed to be named 'A1', 'A2', and 'A3'

    % create score tables for all experiments where result label images are
    % available
    EvaluateResultImagesVsGT(inpath,gtpath);

    % create overall score table, also including experiments where no
    % result mages are available
    CreateOverallFile(inpath,gtpath);
return

function EvaluateResultImagesVsGT(inFolder,gtFolder,saveFolder,username)
% inFolder: folder, where result label images can be found.   
% gtFolder: folder containing testing ground truth label images.
% saveFolder: folder where csv-tables will be stored. Defaults to
% inFolder
%
% Name convention of files to evaluate: 
%   - in the file path or file name a string 'A1', 'A2', or 'A3' needs to 
%     be present exactly once, indicating from which experiment data has 
%     been processed.
%   - the last number in the file name indicates the plant number
%   - results are assumed to be in PNG format with ending '.png'
%
% Name convention of ground truth label images
%   - for each experiment images are in a subfolder 'A1', 'A2', or 'A3'
%   - images are then named plant%03d_label.png such that plant number 7 in
%     experiment A3 has the file name 'A3/plant007_label.png'.
%
% This function is used to evaluate results from the CVPPP 2014 Leaf
% Segmentation Challenge (see http://www.plant-phenotyping.org/CVPPP2014 for 
% more information on the contest). Results are written in separate CSV
% (comma separated value) files names e.g. 'username_A1_results.csv'.
% Please see function 'writeResultTable' for the exact format. 

    if nargin<3
        saveFolder = inFolder;
    end
    if nargin<4
        username = [];
    end


    % check if fromFolder exists, return if not
    if ~isdir(inFolder)
      errorMessage = sprintf('Error: The following folder does not exist:\n%s', inFolder);
      uiwait(warndlg(errorMessage));
      return;
    end

    % recursively loop all subfolders
    filePattern = fullfile(inFolder, '*');
    allFiles = dir(filePattern);
    for k = 1:length(allFiles)
      if allFiles(k).isdir;
         dirName = allFiles(k).name; 
         if( dirName(1) ~= '.')
            fullDirName = fullfile(inFolder, dirName);
            if isempty(username)
                EvaluateResultImagesVsGT(fullDirName,gtFolder,saveFolder,dirName);
            else
                EvaluateResultImagesVsGT(fullDirName,gtFolder,saveFolder,username);
            end
         end
      end
    end

    % write to console what is going on
    fprintf(1, 'Processing %s ...\n', inFolder);

    [theInFiles, imageNumber, experimentNumber] = getExperimentAndImageNumbers(inFolder);

    % calculate scores, also for missing result images: loop through
    % experiments and gt images
    for e = 1:3 % loop experiments

      % check if current experiment shall be processed  
      idx = find(experimentNumber == e,1);
      if ~isempty(idx) 

        [imageNumberGT,bestDiceGT,fgbgDiceGT,absDiffLabelsGT,diffLabelsGT] = getScores(e,gtFolder,inFolder,theInFiles, imageNumber, experimentNumber);  

        resultName = ['A' num2str(e,'%d')];
        writeResultTable(resultName,saveFolder,username,imageNumberGT,bestDiceGT,fgbgDiceGT,absDiffLabelsGT,diffLabelsGT);

      end
    end  
return


% get list of files with their experiment numbers and image numbers for the
% given folder.
function [theInFiles, imageNumber, experimentNumber] = getExperimentAndImageNumbers(inFolder)
% all output variables have the same length being the number of png-files in
% inFolder

    % process all label images from inFolder
    filePattern = fullfile(inFolder, '*.png');
    theInFiles = dir(filePattern);

    % initialize
    N = length(theInFiles); % number of files to check
    imageNumber = zeros(N,1);
    experimentNumber = zeros(N,1);

    % loop result images, get experiment number and image number 
    for k = 1:N

      % get FileName
      baseFileName = theInFiles(k).name; 
      inFileName = fullfile(inFolder, baseFileName);

      % check in which folder we need to look up the ground truth. May be
      % indicated in path or file name, therefore parse inFileName
      if(~isempty(strfind(inFileName,'A1')) || ~isempty(strfind(inFileName,'a1')))
          experimentNumber(k)  = 1;
      elseif(~isempty(strfind(inFileName,'A2')) || ~isempty(strfind(inFileName,'a2')))
          experimentNumber(k)  = 2;
      elseif(~isempty(strfind(inFileName,'A3')) || ~isempty(strfind(inFileName,'a3')))
          experimentNumber(k)  = 3;
      else
          errorMessage = sprintf('Error: Cannot determine folder name for ground truth from:\n %s\n. Aborting!\n\n', inFileName);
          uiwait(warndlg(errorMessage));
          return;
      end

      % parse for last number in file name to determine which label image to
      % compare to
      nums = textscan(baseFileName,'%*[^0123456789]%d');
      imageNumber(k) = nums{end}(end);

    end
return

% calculate scores 
function [imageNumberGT,bestDiceGT,fgbgDiceGT,absDiffLabelsGT,diffLabelsGT] = getScores(e,gtFolder,inFolder,theInFiles, imageNumber, experimentNumber)  
% outputs have all the same length, i.e. the number of gt-files available
% for this experiment

    if nargin < 3
        inFolder = [];
    end
    % determine number of ground truth files for statistics
    subfolderName = ['A' num2str(e,'%d')];
    filePattern = fullfile(gtFolder, subfolderName , '*.png');
    theGtFiles = dir(filePattern);
    M = length(theGtFiles); % number of ground truth files

    % init result tables
    imageNumberGT = zeros(M,1);
    bestDiceGT = zeros(M,1);
    fgbgDiceGT = zeros(M,1);
    absDiffLabelsGT = zeros(M,1);
    diffLabelsGT = zeros(M,1);

    % loop over the gt files
    for k=1:M
        % get FileName and read gt label image
        gtBaseFileName = theGtFiles(k).name; 
        gtFileName = fullfile(  gtFolder, subfolderName, gtBaseFileName);
        gtLabel = imread(gtFileName);
        
        % parse for last number in file name 
        nums = textscan(gtBaseFileName,'%*[^0123456789]%d');
        imageNumberGT(k) = nums{end}(end);

        % print comment
        fprintf(1, 'Processing experiment A%d image number %d ...', e, imageNumberGT(k));

        inLabel = []; % needed to check if inLabel has been created properly
        % check if an input image for imageNumberGT(k) is available
        if ~isempty(inFolder)
            currIdx = find( experimentNumber==e & imageNumber==imageNumberGT(k),1);
            if ~isempty(currIdx)            
                % load available image
                inBaseFileName = theInFiles(currIdx).name; 
                inFileName = fullfile(inFolder, inBaseFileName);
                inLabel = imread(inFileName);
                % if inLabel is not an index image, make it one
                sz = size(inLabel);
                if(length(sz)>2) % inLabel seems to be a color image: convert!
                    % check if grey image in 24bit
                    mx1 = max(abs(inLabel(:,:,1)-inLabel(:,:,2)));
                    mx2 = max(abs(inLabel(:,:,1)-inLabel(:,:,3)));
                    mx = max(mx1(:)+mx2(:));
                    if(mx>0.5) % colored label image
                        [inLabel,map] = rgb2ind(inLabel, 65536,'nodither');
                    else % grey image
                        inLabel = rgb2gray(inLabel);
                    end
                end
            end
        end
        if isempty(inLabel)
            % create an all zero label image
            inLabel = zeros(size(gtLabel));        
            % print warning and continue
            fprintf(1, ' no label image available ...');
        end

        % if size of imLabel is not size of gtLabel, try to interpolate
        if(max(size(inLabel)~=size(gtLabel)))
            fprintf(1, 'wrong size of label image, interpolate  ...');
            inLabel = imresize(inLabel,size(gtLabel),'nearest');
        end        
        
        % calculate performance measures
        bestDiceGT(k) = SymmetricBestDice(inLabel,gtLabel);
        fgbgDiceGT(k) = FGBGDice(inLabel,gtLabel);
        absDiffLabelsGT(k) = AbsDiffFGLabels(inLabel,gtLabel);
        diffLabelsGT(k) = DiffFGLabels(inLabel,gtLabel);
        
        % print comment
        fprintf(1, ' done.\n');
    end
return
    
% write the result tables
function writeResultTable(resultName,saveFolder,username,imageNumberGT,bestDiceGT,fgbgDiceGT,absDiffLabelsGT,diffLabelsGT,experimentNumber)
    
    if nargin < 9
        experimentNumber = []; % needed only, when results from multiple experiments are written in the same table
    end
    
    % now just write the results to a csv table
    resultFileName = fullfile(saveFolder, [username '_' resultName '_results.csv']);
    fprintf(1, 'Writing results to %s\n', resultFileName);
    
    fid = fopen(resultFileName, 'w+');
    if isempty(experimentNumber) % single experiment
        fprintf(fid, 'Results for images: %s\n\n', resultName);
        fprintf(fid, 'number, SymmetricBestDice, FGBGDice, AbsDiffFGLabels, DiffFGLabels\n');    
    else % multiple experiments
        fprintf(fid, 'Results for images: %s\n\n', resultName);
        fprintf(fid, 'number, SymmetricBestDice, FGBGDice, AbsDiffFGLabels, DiffFGLabels, experiment\n');    
    end
    
    M = length(imageNumberGT);
    
    for k=1:M
        if isempty(experimentNumber) % single experiment
            fprintf(fid, '%d, %f, %f, %d, %d\n', imageNumberGT(k),bestDiceGT(k),fgbgDiceGT(k),absDiffLabelsGT(k),diffLabelsGT(k));
        else % multiple experiments
            fprintf(fid, '%d, %f, %f, %d, %d, %d\n', imageNumberGT(k),bestDiceGT(k),fgbgDiceGT(k),absDiffLabelsGT(k),diffLabelsGT(k),experimentNumber(k));
        end
    end
    
    % some statistics...
    fprintf(fid, '\n');
    fprintf(fid, 'mean, %f, %f, %f, %f\n', mean(bestDiceGT),mean(fgbgDiceGT),mean(absDiffLabelsGT),mean(diffLabelsGT));
    fprintf(fid, 'std, %f, %f, %f, %f\n', std(bestDiceGT),std(fgbgDiceGT),std(absDiffLabelsGT),std(diffLabelsGT));
    fprintf(fid, 'median, %f, %f, %f, %f\n', median(bestDiceGT),median(fgbgDiceGT),median(absDiffLabelsGT),median(diffLabelsGT));
    fprintf(fid, 'max, %f, %f, %f, %f\n', max(bestDiceGT),max(fgbgDiceGT),max(absDiffLabelsGT),max(diffLabelsGT));
    fprintf(fid, 'min, %f, %f, %f, %f\n', min(bestDiceGT),min(fgbgDiceGT),min(absDiffLabelsGT),min(diffLabelsGT));

    fclose(fid);
return

% helper function: symmetric best dice
function score = SymmetricBestDice(inLabel,gtLabel)
    score1 = BestDice(inLabel,gtLabel);
    score2 = BestDice(gtLabel,inLabel);
    score = min(score1,score2);
return

% parse written files and create missing ones. Write file containing
% results for all images and all experiments.
function CreateOverallFile(inpath,gtpath)

    % count gt files
    num = zeros(3,1);
    num(1) = getNumberOfFiles(gtpath,'A1');
    num(2) = getNumberOfFiles(gtpath,'A2');
    num(3) = getNumberOfFiles(gtpath,'A3');
    overallNum = sum(num(:));
    % indices of result blocks per experiment
    idx = 1+[0,num(1),num(1)+num(2),num(1)+num(2)+num(3)];
    
    % init score table
    imageNumberAll = zeros(overallNum,1);
    bestDiceAll = zeros(overallNum,1);
    fgbgDiceAll = zeros(overallNum,1);
    absDiffLabelsAll = zeros(overallNum,1);
    diffLabelsAll = zeros(overallNum,1);
    experimentNumberAll = zeros(overallNum,1);
    
    % loop usernames, i.e. directory names
    [dirNames,dirNumber] = getDirNames(inpath);
    for i=1:dirNumber
        for e=1:3 % loop experiments
            % index positions to fill in score table
            pos = idx(e):idx(e+1)-1;
            experimentNumberAll(pos)=e;
            % check if result file for this experiment exists
            resBaseName = [dirNames{i} '_A' num2str(e,'%d') '_results.csv'];
            resFileName = fullfile(inpath, resBaseName);
            r = dir(resFileName);
            if ~isempty(r) % if file exists, process it, else generate results
                 [imageNumberAll(pos),bestDiceAll(pos),fgbgDiceAll(pos),absDiffLabelsAll(pos),diffLabelsAll(pos)] ...
                     = parseResultCSV(resFileName);
            else
                % no results available, generate for label files being zero
                % everywhere
                 [imageNumberAll(pos),bestDiceAll(pos),fgbgDiceAll(pos),absDiffLabelsAll(pos),diffLabelsAll(pos)] ...
                    = getScores(e,gtpath);
                % write score table for this experiment to file
                experimentName = ['A' num2str(e,'%d')];
                writeResultTable(experimentName,inpath,dirNames{i},imageNumberAll(pos),bestDiceAll(pos),fgbgDiceAll(pos),absDiffLabelsAll(pos),diffLabelsAll(pos));
            end
        end
        % write score table for all experiments to file
        writeResultTable('all',inpath,dirNames{i},imageNumberAll,bestDiceAll,fgbgDiceAll,absDiffLabelsAll,diffLabelsAll,experimentNumberAll);
        % write the required LaTeX table
        writeLaTeXTable(inpath,dirNames{i},bestDiceAll,fgbgDiceAll,absDiffLabelsAll,diffLabelsAll,experimentNumberAll);
    end
    
return

% helper function: get number of label files in a directory
function Number = getNumberOfFiles(gtpath,folderName)
    filePattern = fullfile(gtpath, folderName, '*_label.png');
    theFiles = dir(filePattern);
    Number = length(theFiles); % number of ground truth files
return

% get number of directories in a folder, without '.' and '..'
function [DirNames,DirNumber] = getDirNames(inpath)
    filePattern = fullfile(inpath, '*');
    theFiles = dir(filePattern);
    M = length(theFiles); % number of ground truth files
    % count directories without '.' and '..'
    DirNames = cell(M,1);
    DirNumber = 0;
    for k = 1:M
      if theFiles(k).isdir;
        dirName = theFiles(k).name; 
        if dirName(1) ~= '.'            
            DirNumber = DirNumber+1;
            DirNames{DirNumber}=dirName;
        end
      end
    end
return

% parse a result table file and read scores
function [imageNumberGT,bestDiceGT,fgbgDiceGT,absDiffLabelsGT,diffLabelsGT] = parseResultCSV(filename)
    % read file
    fid = fopen(filename);
    fscanf(fid, '%s',9); % skip header
    C = textscan(fid, '%d, %f, %f, %d, %d'); % read data
    fclose(fid);
    % copy read data to output
    imageNumberGT = C{1};
    bestDiceGT = C{2};
    fgbgDiceGT = C{3};
    absDiffLabelsGT = C{4};
    diffLabelsGT = C{5};
return
    
function writeLaTeXTable(saveFolder,username,bestDiceGT,fgbgDiceGT,absDiffLabelsGT,diffLabelsGT,experimentNumber)
    
    % now just write the results to a csv table
    resultFileName = fullfile(saveFolder, [username '_results.tex']);
    fprintf(1, 'Writing results to %s\n', resultFileName);
    
    fid = fopen(resultFileName, 'w+');

    fprintf(fid, '\\begin{tabular}{|l||c|c|c|c|}\n');
    fprintf(fid, '\\hline\n');
    fprintf(fid, ' & \\bf{BestDice [\\%%]} & \\bf{FGBGDice [\\%%]} & \\bf{AbsDiffFGLabels} & \\bf{DiffFGLabels}\\\\\n');
    fprintf(fid, '\\hline\n');
    fprintf(fid, '\\hline\n');

    % mean and std dev for each experiment
    for e=1:3
        idx = find(experimentNumber==e);
        fprintf(fid, '\\bf{A%d} & %.1f ($\\pm$%.1f) & %.1f ($\\pm$%.1f) & %.1f ($\\pm$%.1f) & %.1f ($\\pm$%.1f) \\\\ \n', ...
            e,...
            mean(100*bestDiceGT(idx)),100*std(bestDiceGT(idx)),...
            mean(100*fgbgDiceGT(idx)),100*std(fgbgDiceGT(idx)),...
            mean(absDiffLabelsGT(idx)),std(absDiffLabelsGT(idx)),...
            mean(diffLabelsGT(idx)),std(diffLabelsGT(idx)));
        fprintf(fid, '\\hline\n');
    end
    % overall mean and std dev
        fprintf(fid, '\\bf{all} & %.1f ($\\pm$%.1f) & %.1f ($\\pm$%.1f) & %.1f ($\\pm$%.1f) & %.1f ($\\pm$%.1f) \\\\ \n', ...
        mean(100*bestDiceGT),100*std(bestDiceGT),...
        mean(100*fgbgDiceGT),100*std(fgbgDiceGT),...
        mean(absDiffLabelsGT),std(absDiffLabelsGT),...
        mean(diffLabelsGT),std(diffLabelsGT));
    fprintf(fid, '\\hline\n');
    fprintf(fid, '\\end{tabular}\n');
    
    fclose(fid);
return