# Data Formatting Docs
To keep this repo lightweight, datasets will be selectively downloaded by each notebook. All other files in this directory will be ignored by Git.

## mnist.pkl.gz
This is the popular mnist dataset. This file contains 70000 tuples of the form `(image, label)`. The images are NumPy arrays of shape `(784,)`, which represent 28x28 images. The values are floats between 0 and 1. The labels are one-hot vectorized representations of the digits, with shape `(10,)`. In a tuple whose image depicts a 0, the label would look like `np.array([1, 0, 0 ...])`.

To load the data, use:
```python
dir_path = os.path.dirname(os.path.realpath('__file__')) # absolute path of notebook
dataset_path = os.path.join(dir_path, "../data/mnist.pkl.gz")
if not os.path.exists(dataset_path):
    print('Downloading dataset with wget module...')
    if not os.path.exists(os.path.join(dir_path, '../data')):
        os.mkdir(os.path.join(dir_path, '../data'))
    url = 'http://ericjmichaud.com/downloads/mnist.pkl.gz'
    wget.download(url, dataset_path)  
print('Download failed') if not os.path.exists(dataset_path) else print('Dataset acquired')
f = gzip.open(dataset_path, 'rb')
mnist = pickle.load(f)
f.close()
print('Loaded data to variable `mnist`')
```