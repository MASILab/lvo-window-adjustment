import numpy as np
from PIL import Image
import pandas as pd
import json
import os
import glob

def get_sub_info_fromcsv( data_dir, image_out_dir, test_list, data_list_file, split):
    json_dict = []

    type_dict = {'normal' : 'normal',
                 'Obsolescent': 'obsolescent', 'obsolescent': 'obsolescent',
                 'solidified': 'solidified',
                 'dispear':'disappearing', 'disppear':'disappearing', 'disppeared':'disappearing', 'disppearing':'disappearing',
                 'unknown' : 'unknown', 'Description':'unknown',
                 }
    clss_dict = {'normal': 0,
                 'obsolescent': 1,
                 'solidified': 2,
                 'disappearing': 3,
                 'unknown': 4,
                 }

    unique_subs = os.listdir(data_dir)
    unique_subs.sort()
    for si in range(len(unique_subs)):
        one_dict = {}

        image_file_name = unique_subs[si]
        image_name_strs = image_file_name.split('-x-')

        image_name = image_name_strs[0]
        image_label = image_name_strs[2]

        if split == 'test':
            if not image_name_strs[0] in test_list:
                continue
        else:
            if image_name_strs[0] in test_list:
                continue

        image_path = os.path.join(data_dir, image_file_name)

        try:
            pathology_name = type_dict[image_label]
            pathology_num = clss_dict[pathology_name]

            image_data_dir = os.path.join(image_out_dir, split, pathology_name)
        except:
            aaa= 1
        if not os.path.exists(image_data_dir):
            os.makedirs(image_data_dir)
        image_data_file = os.path.join(image_data_dir, os.path.basename(image_path))
        if not os.path.exists(image_data_file):
            os.system('cp "%s" "%s"'%(image_path, image_data_file))

        one_dict['name'] = image_name
        one_dict['image_path'] = image_data_file
        one_dict['pathology'] = pathology_name
        one_dict['label'] = pathology_num
        json_dict.append(one_dict)

    clss_count = {'normal': 0,
                 'obsolescent': 0,
                 'solidified': 0,
                 'disappearing': 0,
                 'unknown': 0}

    for ji in range(len(json_dict)):
        clss_count[json_dict[ji]['pathology']] = clss_count[json_dict[ji]['pathology']]+1

    print('========%s=========' % split)
    print(clss_count)
    with open(data_list_file, 'w') as f:
        json.dump(json_dict, f)

    return json_dict


if __name__ == "__main__":
    known_dir = '/Users/zhende-MAC/Documents/GitHub/mixmatch_data/R24/image_iso'
    unknown_dir = '/Users/zhende-MAC/Documents/GitHub/mixmatch_data/batch1/image_iso'

    known_train_json = '/Users/zhende-MAC/Documents/GitHub/json/known_train.json'
    known_test_json = '/Users/zhende-MAC/Documents/GitHub/json/known_test.json'
    unknown_train_json = '/Users/zhende-MAC/Documents/GitHub/json/unknown_train.json'

    image_out_dir = '/Users/zhende-MAC/Documents/GitHub/mixmatch_output_data'

    test_list = ['22861_2017-04-08 12_12_09','23899_2017-04-07 22_56_53','23900_2017-04-07 23_07_54','23901_2017-04-07 23_23_05','24369_2017-04-08 04_54_25']

    get_sub_info_fromcsv(known_dir, image_out_dir, test_list, known_test_json, 'test')
    get_sub_info_fromcsv(known_dir, image_out_dir, test_list, known_train_json, 'train')
    get_sub_info_fromcsv(unknown_dir, image_out_dir, [], unknown_train_json, 'train')


