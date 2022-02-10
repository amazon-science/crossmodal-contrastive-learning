import argparse
from copy import deepcopy
import itertools
import json
import os
from pathlib import Path

import h5py
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from utils_collection.text_embedding import preprocess_bert_paragraph


def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument("dataset_name", type=str, default="default", help="dataset name")
    parser.add_argument("--dataroot",
                        type=str,
                        default="data",
                        help="change default path to dataset")
    parser.add_argument("--modality",
                        type=str,
                        default="howto100m_finetune",
                        help="select the modality name")
    parser.add_argument("--group_k",
                        type=int,
                        default=5,
                        help="Number of segments per video")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--bert_cache_path",
                        type=str,
                        default="pretrained_bert_model",
                        help="batch size")
    args = parser.parse_args()
    dataset_path = args.dataroot #/ args.dataset_name

    # setup paths
    meta_file = Path(os.path.join(dataset_path, "meta", "meta_group{}_{}.json".format(args.group_k, args.modality)))
    print("loading meta file: ", meta_file)
    lengths_file = Path(os.path.join(dataset_path, "group{}".format(args.group_k), "language_features", "text_lens_default.json"))
    data_file = Path(os.path.join(dataset_path, "group{}".format(args.group_k), "language_features", "text_default.h5"))
    print(data_file)
    if data_file.exists():
        print(f"{data_file} already exists. nothing to do.")
        return

    # load pretrained bert model
    print("load bert model...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",
                                              cache_dir=args.bert_cache_path)
    model = BertModel.from_pretrained("bert-base-uncased",
                                      output_hidden_states=True,
                                      cache_dir=args.bert_cache_path)
    if args.cuda:
        model = model.cuda()

    # load metadata
    vids_dict = json.load(meta_file.open("rt", encoding="utf8"))
    layer_list_int = [-2, -1]

    # loop videos and encode features
    data_h5 = h5py.File(data_file, "w")
    lengths = {}
    for vid_id, meta in vids_dict.items(): #, desc="compute text features"):
        # collect narrations and preprocess them for bert
        # print(meta, vid_id)
        sentences = [seg["narration"] for seg in meta["segments"]]
        paragraph = preprocess_bert_paragraph(sentences)
        # tokenize
        sent_tokens = []
        total_len = 0
        for list_of_words in paragraph:
            sentence = " ".join(list_of_words)
            sentence_int_tokens = tokenizer.encode(sentence,
                                                   add_special_tokens=False)
            sent_tokens.append(sentence_int_tokens)
            total_len += len(sentence_int_tokens)
        if total_len > 512:
            # cut too long tokens if needed, retaining some parts for all
            # sentences and all the SEP tokens
            s_caps_lens_old = [len(sentence) for sentence in sent_tokens]
            s_cap_lens = deepcopy(s_caps_lens_old)
            min_cut = 5
            for sent in range(len(paragraph) - 1, -1, -1):
                overshoot = sum(s_cap_lens) - 512
                if overshoot == 0:
                    break
                new_len = max(min_cut, len(sent_tokens[sent]) - overshoot)
                s_cap_lens[sent] = new_len
            sent_tokens_new = []
            for i, (old_len,
                    new_len) in enumerate(zip(s_caps_lens_old, s_cap_lens)):
                if old_len != new_len:
                    sent_tokens_new.append(sent_tokens[i][:new_len - 1] +
                                           [102])
                else:
                    sent_tokens_new.append(sent_tokens[i])
            sent_tokens = sent_tokens_new

        # encode with bert
        sent_lens = [len(sentence) for sentence in sent_tokens]
        flat_paragraph = list(itertools.chain.from_iterable(sent_tokens))
        input_tensor = torch.tensor(flat_paragraph).long().unsqueeze(0)
        if args.cuda:
            input_tensor = input_tensor.cuda()
        _, _, layer_output = model(input_tensor)
        features = []
        for layer_num in layer_list_int:
            layer_features = layer_output[layer_num].squeeze(0)
            features.append(layer_features)
        features = torch.cat(features, -1)
        assert features.shape[0] == sum(sent_lens), ""

        # write features
        data_h5[vid_id] = features.detach().cpu().numpy()
        lengths[vid_id] = sent_lens
    data_h5.close()

    # write lengths
    json.dump(lengths, lengths_file.open("wt", encoding="utf8"))


if __name__ == '__main__':
    main()
