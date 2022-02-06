==================================================================
Large Scale Movie Description and Understanding Challenge (LSMDC), at ECCV 2016
==================================================================
Movie Annotation and Retrieval
==================================================================

Get the Linux/Mac download script (downloadChallengeDataMAR.sh) and 
copy it to the location where you want to save the files and then run:
  downloadChallengeDataMAR.sh <username-MPIIMD> <password-MPIIMD>
  
Note: Instructions how to obtain the username/password information 
are here: https://sites.google.com/site/describingmovies/lsmdc-2016/download

In the following:
M-VAD: Montreal Video Annotation Dataset [1]
MPII-MD: MPII Movie Description dataset [2]

==================================================================
Annotations
==================================================================

= Files
- Training: LSMDC16_paraphrased_train_subset.txt
- Validation: None (original sentences from movie description could be used)
- Public test: LSMDC16_paraphrased_test.txt

= Format
- Each line of the annotation *.csv file contains:
  - Each line of the annotation *.csv file contains:
   <CLIP_ID>\t<START_ALIGNED>\t<END_ALIGNED>\t<START_EXTRACTED>\t<END_EXTRACTED>\t<SENTENCE>\t<PARAPHRASED SENTENCE>
   where "\t" is a TAB character, <START_*> and <END_*> are time-stamps "hh.mm.ss.msmsms" (e.g. 01.02.27.034).
  Note, that in case where the manually aligned video clip is shorter than 2 seconds, we symmetrically 
  expand it (from beginning and end) to be exactly 2 seconds long. Thus, <START_ALIGNED> and <END_ALIGNED>
  correspond to precise manually obtained time-stamps, while <START_EXTRACTED>, <END_EXTRACTED> indicate
  the actual extracted clip's start and end.
- <SENTENCE> is a complete reference sentence
- <PARAPHRASED SENTENCE> is a sentence that is the summarized or main aspect of what is described in the original sentence containing 3-10 words
- The task is to train a model for video annotation and retrieval.

= Statistics
- Training: 19,948 (subset of training data "sentences longer than ~15 words" )
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
