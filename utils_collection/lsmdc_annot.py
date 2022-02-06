import re
import json
import argparse
import os
import csv

import numpy as np
import h5py

# Modified from https://github.com/jamespark3922/lsmdc-baseline/blob/master
def build_vocab(params):
    count_thr = params['word_count_threshold']
    # count up the number of words
    counts = {}
    max_len = []
    csvs = params['annot'] #['LSMDC16_annos_training.csv', 'LSMDC16_annos_val.csv', 'LSMDC16_annos_test.csv']
    for c in csvs:
        with open(os.path.join(params['input_path'], c)) as csv_file:
            csv_reader = csv.reader(csv_file,delimiter='\t')
            for row in csv_reader:
                # remove punctuation but keep possessive because we want to separate out character names
                ws = re.sub(r'[.!,;?]', ' ', str(row[5]).lower()).replace("'s", " 's").split()
                max_len.append(len(ws))
                for w in ws:
                    counts[w] = counts.get(w, 0) + 1
    print('max count', np.mean(max_len))
    # cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    total_words = sum(counts.values())
    bad_words = [w for w, n in counts.items() if n <= count_thr]
    vocab = [w for w, n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' %
          (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
    print('number of words in vocab would be %d' % (len(vocab), ))
    print('number of UNKs: %d/%d = %.2f%%' %
          (bad_count, total_words, bad_count * 100.0 / total_words))
    # lets now produce the final annotations
    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        print('inserting the special UNK token')
        vocab.append('<UNK>')

    splits = params['splits'] #['train', 'val', 'test', 'blind_test']
    videos = []
    movie_ids = {}
    vid = 0
    groups = []
    gid = -1
    for i,c in enumerate(csvs):
        split = splits[i]
        with open(os.path.join(params['input_path'], c)) as csv_file:
            csv_reader = csv.reader(csv_file,delimiter='\t')
            for row in csv_reader:
                clip = row[0]
                movie = clip[:clip.rfind('_')]
                info = {'id': vid, 'split' : split, 'movie' : movie, 'clip' : clip}
                if movie not in movie_ids:
                    gid+=1
                    ginfo = {'id': gid, 'split': split, 'movie': movie, 'videos': [vid]}
                    groups.append(ginfo)
                    gcount = 0
                    movie_ids[movie] = [gid]
                else:
                    if gcount >= params['group_by']:
                        gid+=1
                        ginfo = {'id': gid, 'split': split, 'movie': movie, 'videos': [vid]}
                        groups.append(ginfo)
                        gcount = 0
                        movie_ids[movie].append(gid)
                    else:
                        groups[gid]['videos'].append(vid)
                if split != 'blind_test':
                    ws = re.sub(r'[.!,;?]', ' ', str(row[5]).lower()).replace("'s", " 's").split()
                    #print(ws)
                    #print(" ".join(ws))
                    #print(row[5])
                    #print('=='*10)
                    caption = ['<eos>'] + [w if counts.get(w, 0) > count_thr else '<UNK>' for w in ws] + ['<eos>']
                    info['final_caption'] = caption
                    info['narration'] = " ".join(ws)
                videos.append(info)
                vid+=1
                gcount+=1
    return videos, groups, movie_ids, vocab


def build_label(params, videos, wtoi):
    max_length = params['max_length']
    N = len(videos)

    label_arrays = []
    label_lengths = np.zeros(N, dtype='uint32')
    bt = 0
    for i, video in enumerate(videos):
        if 'final_caption' not in video:
            bt+=1
            continue
        s = video['final_caption']
        assert len(s) > 0, 'error: some video has no captions'

        Li = np.zeros((max_length), dtype='uint32')
        label_lengths[i] = min(max_length, len(s))  # record the length of this sequence
        for k, w in enumerate(s):
            if k < max_length - 1:
                Li[k] = wtoi[w]

        # note: word indices are 1-indexed, and captions are padded with zeros
        label_arrays.append(Li)
    total = N - bt
    labels = np.array(label_arrays)[:total]  # put all the labels together
    label_lengths = label_lengths[:total]
    assert labels.shape[0] == total, 'lengths don\'t match? that\'s weird'
    assert labels[:,-1].sum() == 0 , 'sequences do not end on <eos>'
    assert np.all(label_lengths > 2), 'error: some caption had no words?'

    print('encoded captions to array of size ', labels.shape)
    return labels, label_lengths