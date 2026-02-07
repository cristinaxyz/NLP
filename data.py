from datasets import load_dataset

ds = load_dataset("sh0416/ag_news")

seed = 42

split = ds['train'].train_test_split(test_size = 0.1, seed = seed)

training_ds = split['train']
dev_ds = split['test']
test_ds = ds['test']

print(len(training_ds))
print(len(dev_ds))
print(len(test_ds))