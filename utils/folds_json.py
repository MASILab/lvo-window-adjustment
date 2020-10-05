import os
import pandas
import json
import shutil
import numpy as np


def get_csv_info(sub_csv, json_dir, image_dir, image_out_dir, split):
    json_list = list()
    df = pandas.read_csv(sub_csv)
    for row in df.itertuples(index=False):
        image = row[0]
        condition = row[1]
        label = row[2]

        image_name = image.split('-x-')[0]

        image_path = os.path.join(image_out_dir, split, condition, image)

        if label != -1:
            one_dict = dict()
            one_dict['name'] = image_name
            one_dict['image_path'] = image_path
            one_dict['pathology'] = condition
            one_dict['label'] = label
            json_list.append(one_dict)

        input_dir = os.path.join(image_dir, image)
        output_dir = os.path.join(image_out_dir, split, condition)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        image_data_file = os.path.join(output_dir, image)
        if not os.path.exists(image_data_file):
            os.system('cp "%s" "%s"' % (input_dir, image_data_file))

    with open(json_dir, 'w') as f:
        json.dump(json_list, f)

    return json_list


if __name__ == "__main__":
    known_dir = '/Data/luy8/GitHub/resized_image'

    fold1_csv = '/Data/luy8/GitHub/folds/fold_1.csv'
    fold2_csv = '/Data/luy8/GitHub/folds/fold_2.csv'
    fold3_csv = '/Data/luy8/GitHub/folds/fold_3.csv'
    fold4_csv = '/Data/luy8/GitHub/folds/fold_4.csv'
    fold5_csv = '/Data/luy8/GitHub/folds/fold_5.csv'

    fold1_json = '/Data/luy8/GitHub/json/fold_1.json'
    fold2_json = '/Data/luy8/GitHub/json/fold_2.json'
    fold3_json = '/Data/luy8/GitHub/json/fold_3.json'
    fold4_json = '/Data/luy8/GitHub/json/fold_4.json'
    fold5_json = '/Data/luy8/GitHub/json/fold_5.json'

    image_out_dir = '/Data/luy8/GitHub/prepared_data'

    get_csv_info(fold1_csv, fold1_json, known_dir, image_out_dir, 'fold1')
    get_csv_info(fold2_csv, fold2_json, known_dir, image_out_dir, 'fold2')
    get_csv_info(fold3_csv, fold3_json, known_dir, image_out_dir, 'fold3')
    get_csv_info(fold4_csv, fold4_json, known_dir, image_out_dir, 'fold4')
    get_csv_info(fold5_csv, fold5_json, known_dir, image_out_dir, 'fold5')

    json_list = [fold1_json, fold2_json, fold3_json, fold4_json, fold5_json]
    json_dir = os.path.dirname(json_list[0])
    json_list_dir = []
    for fold in json_list:
        json_list_dir.append(os.path.join(json_dir, fold))

    for k in json_list_dir:
        # make folder
        set_name = os.path.basename(k).split('.')[0]
        set_folder = os.path.join(json_dir, set_name)
        if not os.path.exists(set_folder):
            os.makedirs(set_folder)

        # select test set
        test_file = os.path.join(set_folder, 'test.json')
        shutil.copy(k, test_file)
        folds_list_copy = json_list_dir.copy()
        folds_list_copy.remove(k)

        # select val set
        val_set = folds_list_copy[np.random.randint(0, len(folds_list_copy))]
        val_file = os.path.join(set_folder, 'validate.json')
        shutil.copy(val_set, val_file)
        folds_list_copy.remove(val_set)

        # combine train sets
        train_set = []
        for file in folds_list_copy:
            with open(file) as json_file:
                data = json.load(json_file)
            train_set = train_set + data
        train_file = os.path.join(set_folder, 'train.json')
        with open(train_file, 'w') as f:
            json.dump(train_set, f)
