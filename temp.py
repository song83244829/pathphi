from datasets import load_dataset
eval_size = 500
eval_dataset = load_dataset(
    'HuggingFaceM4/the_cauldron', 'nlvr2', split=f'train[:{eval_size}]', num_proc=8, cache_dir="/scratch/09697/luosong/cache"
)
