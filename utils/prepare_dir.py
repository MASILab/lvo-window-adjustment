import os
import pandas
import json


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

    train_csv = '/Data/luy8/GitHub/train.csv'
    validation_csv = '/Data/luy8/GitHub/validate.csv'

    known_train_json = '/Data/luy8/GitHub/json/known_train.json'
    known_validate_json = '/Data/luy8/GitHub/json/known_validate.json'

    image_out_dir = '/Data/luy8/GitHub/prepared_data'

    get_csv_info(train_csv, known_train_json, known_dir, image_out_dir, 'train')
    get_csv_info(validation_csv, known_validate_json, known_dir, image_out_dir, 'validate')
