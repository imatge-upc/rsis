%   Author: Hanno Scharr, Institute of Bio- and Geosciences II,
%   Forschungszentrum Jülich
%   Contact: h.scharr@fz-juelich.de
%   Date: 02.02.2016
%
% Copyright 2016, Forschungszentrum Jülich GmbH, 52425 Jülich, Germany
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
 

function score = BestScore(inLabel,gtLabel,ScoreFun)
% inLabel: label image to be evaluated. Labels are assumed to be consecutive numbers.
% gtLabel: ground truth label image. Labels are assumed to be consecutive numbers.
% score: max value delivered by ScoreFun
%
% We assume that the lowest label in inLabel is background, same for gtLabel
% and do not use it. This is necessary to avoid that the trivial solution, 
% i.e. finding only background, gives excellent results.
%
% For the original Dice score or Overlap score, labels corresponding to each other need to
% be known in advance. Here we simply take the best matching label from 
% gtLabel in each comparison. We do not make sure that a label from gtLabel
% is used only once. Better measures may exist. 
% Please enlighten me if I do something stupid here...

score = 0; % initialize output

% check if label images have same size
if(max(size(inLabel)~=size(gtLabel)))
    return
end

maxInLabel = max(inLabel(:)); % maximum label value in inLabel
minInLabel = min(inLabel(:)); % minimum label value in inLabel
maxGtLabel = max(gtLabel(:)); % maximum label value in gtLabel
minGtLabel = min(gtLabel(:)); % minimum label value in gtLabel

if(maxInLabel==minInLabel) % trivial solution
    return
end

for i=[minInLabel+1:maxInLabel]; % loop all labels of inLabel, but background
    sMax = 0; % maximum score value found for label i so far
    for j=[minGtLabel+1:maxGtLabel] % loop all labels of gtLabel, but background
        % calculate Dice score for the given labels i and j
        inMask = (inLabel==i); % find region of label i in inLabel
        gtMask = (gtLabel==j); % find region of label j in gtLabel
        s = ScoreFun(inMask,gtMask); % compare labelled regions
        % keep max score value for label i
        if(sMax < s)
            sMax = s;
        end
    end
    score = score + sMax; % sum up best found values
end
score = score/double(maxInLabel-minInLabel);






