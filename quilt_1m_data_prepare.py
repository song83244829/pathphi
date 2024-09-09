"""
modified from https://huggingface.co/docs/transformers/tasks/video_classification
and https://huggingface.co/docs/transformers/main/en/model_doc/video_llava
"""
import os
import dask.dataframe as dd
import pathlib
import json

# Step 3: Define a function to process each row
def process_row(row):
    prompt = 'You are anatomic pathologist. try to use your knowledge describe the image I provided using medical terms.'
    image_path = row['image_path']
    split = row['split']
    i = row.iloc[0]
    caption = row['caption']
    split_dict = {
                    'id': f'{split}-{i:010d}',
                    'source': 'quilt_1m',
                    'conversations': [
                        {
                            'images': image_path,
                            'user': prompt,
                            'assistant': caption,
                        }
                    ],
                }
        
    return split_dict


def main(dataset_dir):
    dataset_dir = "/scratch/09697/luosong/databases/quilt_1M"
    df = dd.read_csv(os.path.join(dataset_dir, "quilt_1M_lookup.csv"))
    # Step 4: Apply the function to each partition and convert to list of dicts
    result = df.map_partitions(lambda df: df.apply(process_row, axis=1)).compute()
    result = result.tolist()

    split2examples = {'train': [], 'val': []}
    for i in result:
        if i['id'].startswith('train'):
            split2examples['train'].append(i)
        if i['id'].startswith('val'):
            split2examples['val'].append(i)

    for split, examples in split2examples.items():
        with open(dataset_dir + f'/quilt_1m_{split}.jsonl', 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')


if __name__ == '__main__':
    dataset_folder = "/scratch/09697/luosong/databases/quilt_1M"
    main(dataset_folder)