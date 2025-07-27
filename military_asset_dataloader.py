import kagglehub
from datasets import load_dataset, Dataset


def sample_images(split_size=3):
    input_dataset = get_dataset(split_name='test')
    input_dataset = sample_dataset(input_dataset, split_size)
    return input_dataset['image'] 

def get_dataset(split_name):
    path = kagglehub.dataset_download("rawsi18/military-assets-dataset-12-classes-yolo8-format")

    split = split_name # can be set to train or val as well
    path_subset = path + "\\military_object_dataset\\" + split

    image_dataset = load_dataset(path_subset + "\\images")['train']
    label_dataset = load_dataset("text", data_dir=path_subset + "\\labels", sample_by="document")['train']

    data = {
        'image': image_dataset['image'],
        'text': label_dataset['text'] 
    }

    ds = Dataset.from_dict(data)

    return ds
    

def sample_dataset(ds, split_size):
    return_ds = ds.shuffle(seed=split_size)
    return_ds = return_ds.select(range(split_size))
    return return_ds

def get_images(split='test', start=0, end=None):
    dataset = get_dataset(split)

    if end == None:
        return dataset['image']
    
    return_ds = dataset.select(range(start, end))
    return return_ds['image']