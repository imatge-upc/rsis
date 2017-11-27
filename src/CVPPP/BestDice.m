%   Author: Hanno Scharr, Institute of Bio- and Geosciences II,
%   Forschungszentrum Jülich
%   Contact: h.scharr@fz-juelich.de
%   Date: 17.3.2014
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
 

function score = BestDice(inLabel,gtLabel)
% inLabel: label image to be evaluated. Labels are assumed to be consecutive numbers.
% gtLabel: ground truth label image. Labels are assumed to be consecutive numbers.
% score: Dice score
%
% For the original Dice score, labels corresponding to each other need to
% be known in advance. Here we simply take the best matching label from 
% gtLabel in each comparison. We do not make sure that a label from gtLabel
% is used only once. Better measures may exist. Please enlighten me if I do
% something stupid here...

% check if label images have same size
if(max(size(inLabel)~=size(gtLabel)))
    score = 0;
    return
end

maxInLabel = max(inLabel(:)); % maximum label value in inLabel
minInLabel = min(inLabel(:)); % minimum label value in inLabel
maxGtLabel = max(gtLabel(:)); % maximum label value in gtLabel
minGtLabel = min(gtLabel(:)); % minimum label value in gtLabel

score = 0; % initialize output
for i=[minInLabel:maxInLabel]; % loop all labels of inLabel
    sMax = 0; % maximum Dice value found for label i so far
    for j=[minGtLabel:maxGtLabel] % loop all labels of gtLabel
        s = Dice(inLabel, gtLabel, i, j); % compare labelled regions
        % keep max Dice value for label i
        if(sMax < s)
            sMax = s;
        end
    end
    score = score + sMax; % sum up best found values
end
score = score/double(maxInLabel-minInLabel+1);

function out = Dice(inLabel, gtLabel, i, j)
% calculate Dice score for the given labels i and j

inMask = (inLabel==i); % find region of label i in inLabel
gtMask = (gtLabel==j); % find region of label j in gtLabel
inSize = sum(inMask(:)); % cardinality of set i in inLabel
gtSize = sum(gtMask(:)); % cardinality of set j in gtLabel
overlap= sum(inMask(:).*gtMask(:)); % cardinality of overlap of the two regions
out = 2*overlap/(inSize + gtSize); % Dice score



