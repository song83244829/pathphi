"""
modified from https://huggingface.co/docs/transformers/tasks/video_classification
and https://huggingface.co/docs/transformers/main/en/model_doc/video_llava
"""
import os
import json
import pathlib
import shutil
import dask.dataframe as dd


def main(dataset_dir):
        # Read the CSV file in parallel
    df = dd.read_csv(os.path.join(dataset_dir, "quilt_1M_lookup.csv"))

    class_labels = sorted({str(path).split('/')[-2] for path in all_video_file_paths})
    prompt = f'Classify the video into one of the following classes: {", ".join(class_labels)}.'

    # convert all videos
    split2examples = {'train': [], 'val': [], 'test': []}
    out_path = pathlib.Path(out_dir)
    out_image_path = out_path / 'images'
    for i, video_file_path in enumerate(all_video_file_paths):
        # get train/val/test
        split = video_file_path.parts[-3]
        label = video_file_path.parts[-2]
        images = video_to_images(video_file_path)

        image_path_prefix = '/'.join(video_file_path.with_suffix('').parts[-3:])
        split2examples[split].append(
            {
                'id': f'{split}-{i:010d}',
                'source': 'ucf101',
                'conversations': [
                    {
                        'images': [
                            f'{image_path_prefix}.{i}.jpg' for i in range(len(images))
                        ],
                        'user': prompt,
                        'assistant': label,
                    }
                ],
            }
        )
        (out_image_path / image_path_prefix).parent.mkdir(parents=True, exist_ok=True)
        for i, image in enumerate(images):
            image.save((out_image_path / image_path_prefix).with_suffix(f'.{i}.jpg'))

    for split, examples in split2examples.items():
        with open(out_path / f'ucf101_{split}.jsonl', 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')

    # remove tmp_path recursively
    shutil.rmtree(tmp_path)


if __name__ == '__main__':
    dataset_folder = "/scratch/09697/luosong/databases/quilt_1M"
    main(dataset_folder)