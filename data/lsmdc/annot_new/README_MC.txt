==================================================================
Large Scale Movie Description and Understanding Challenge (LSMDC), at ECCV 2016
==================================================================
Movie Multiple-Choice Test
==================================================================

Get the Linux/Mac download script (downloadChallengeDataMC.sh) and 
copy it to the location where you want to save the files and then run:
  downloadChallengeDataMC.sh <username-MPIIMD> <password-MPIIMD>
  
Note: Instructions how to obtain the username/password information 
are here: https://sites.google.com/site/describingmovies/lsmdc-2016/download

In the following:
M-VAD: Montreal Video Annotation Dataset [1]
MPII-MD: MPII Movie Description dataset [2]

==================================================================
Annotations
==================================================================

= Files
- Training: LSMDC16_multiple_choice_train.csv
- Validation: LSMDC16_multiple_choice_valid.csv
- Public test: LSMDC16_multiple_choice_test_randomized.csv

= Format
-
- Each line of the annotation *.csv file contains:
  - For LSMDC16_multiple_choice_train.csv and  LSMDC16_multiple_choice_valid.csv:
   <CLIP_ID>\t<START_ALIGNED>\t<END_ALIGNED>\t<START_EXTRACTED>\t<END_EXTRACTED>\t<CORRECT_SENTENCE>\t<DISTRACTOR_SENTENCE1>..
   \t<DISTRACTOR_SENTENCE2>\t<DISTRACTOR_SENTENCE3>\t<DISTRACTOR_SENTENCE4>
   
  - For LSMDC16_multiple_choice_test_randomized.csv:
   <CLIP_ID>\t<START_ALIGNED>\t<END_ALIGNED>\t<START_EXTRACTED>\t<END_EXTRACTED>\t<SENTENCE1>\t<SENTENCE2>..
   \t<SENTENCE3>\t<SENTENCE4>\t<SENTENCE5>\t<ANSWER_INDEX(between 1-5)> 
   
  where "\t" is a TAB character, <START_*> and <END_*> are time-stamps "hh.mm.ss.msmsms" (e.g. 01.02.27.034).
  Note, that in case where the manually aligned video clip is shorter than 2 seconds, we symmetrically 
  expand it (from beginning and end) to be exactly 2 seconds long. Thus, <START_ALIGNED> and <END_ALIGNED>
  correspond to precise manually obtained time-stamps, while <START_EXTRACTED>, <END_EXTRACTED> indicate
  the actual extracted clip's start and end.

- <CORRECT_SENTENCE> is ground-truth caption or annotation for corresponding video
- <DISTRACTOR_SENTENCE> is a distractor annotation that randomly have been picked from the corpus with the condition that they are labeled with different activity-phrase label than the correct answer. 
However objects, location, some context might be the similar in both correct sentence and distractor sentences. But the human activities are organized in the way to be different from correct answer
as much as possible.
- <SENTENCE> in public test could be either CORRECT_SENTENCE or DISTRACTOR_SENTENCE
- <ANSWER_INDEX> in public test in index of correct answer which is  a number between 1 to 5 
  
- The task is multiple-choice test: given a video query and 5 sentence choices, retrieve the correct sentence for the video among 5 answer choices (i.e. 1 correct answer and 4 distractor captions) 

= Statistics
- Training: 101,079
- validation: 7,408
- Public Test: 10,053 

==================================================================
[1]
@article{AtorabiM-VAD2015,
author = {Torabi, Atousa and Pal, Chris and Larochelle, Hugo and Courville, Aaron},
title = {Using Descriptive Video Services To Create a Large Data Source For Video Annotation Research},
journal = {arXiv preprint},
year = {2015},
url = {http://arxiv.org/pdf/1503.01070v1.pdf}}

[2]
@inproceedings{rohrbach15cvpr,
title={A Dataset for Movie Description},
author={Rohrbach, Anna and Rohrbach, Marcus and Tandon, Niket and Schiele, Bernt},
booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
url = {http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Rohrbach_A_Dataset_for_2015_CVPR_paper.pdf}
year={2015}}