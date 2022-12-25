import torch
import cv2
from annoy import AnnoyIndex
import random
import albumentations as albu
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from processing.data_processing import get_train_test_df, split_data


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

data_dir = '../../../../Downloads/Stanford_Online_Products/'
_, test_df = get_train_test_df(data_dir)
X_train, X_val = split_data(data_dir)

# Data augmentation
augmenter = albu.Compose([
    albu.HorizontalFlip(),
    albu.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.1, rotate_limit=10, p=0.4),
    albu.OneOf(
        [
            albu.RandomBrightnessContrast(),
            albu.RandomGamma(),
            albu.MedianBlur(),
        ],
        p=0.5
    ),
])


def load_img_from_path(path):
    img = cv2.imread(data_dir + path)
    # (250, 250) image size
    img = cv2.resize(img, (250, 250), interpolation=cv2.INTER_NEAREST)

    img = augmenter(image=img)['image']
    img = np.float32(img.transpose((2, 0, 1)) / 255)
    return img


def get_vector_for_img(img, model):
    layer = model._modules.get('avgpool')
    feature_vector = torch.zeros(512)

    def copy_layer_output(m, i, o):
        feature_vector.copy_(o.data.reshape(o.data.size(1)))

    h = layer.register_forward_hook(copy_layer_output)
    model(torch.tensor(img).unsqueeze(0).to(device))
    h.remove()

    return feature_vector


def build_index(model, title):
    f = 512  # Length of item vector that will be indexed
    t = AnnoyIndex(f, 'angular')
    for i in tqdm(range(len(X_train))):
      img = load_img_from_path(X_train.iloc[i]['path'])
      v = get_vector_for_img(img, model)
      t.add_item(i, v)

    t.build(10) # 10 trees
    t.save(f'{title}.ann')
    return t


def visualize_retrieval(model, annoy_index, title):
    plt.rcParams['figure.figsize'] = [20, 10]
    rows, cols = 5, 5
    f, axarr = plt.subplots(rows, cols)

    for row in range(rows):
        # get random image
        i = random.randint(0, 60502)
        img = load_img_from_path(test_df.iloc[i]['path'])
        super_class_id = test_df.iloc[i]['super_class_id']

        # plot this image
        axarr[row, 0].imshow(np.transpose(img, (1, 2, 0)))
        axarr[row, 0].axis('off')
        axarr[row, 0].set_title(f'Super Class: {super_class_id}')

        # get retrieval images
        feature_v = get_vector_for_img(img, model)
        neightbours = annoy_index.get_nns_by_vector(feature_v, 4)

        # plot retrieval images
        for col in range(1, cols):
            axarr[row, col].axis('off')
            retrieval_img = load_img_from_path(test_df.iloc[neightbours[col-1]]['path'])
            retrieval_super_class_id = test_df.iloc[neightbours[col-1]]['super_class_id']
            axarr[row, col].imshow(np.transpose(retrieval_img, (1, 2, 0)))
            axarr[row, col].set_title(f'Retrieval Super Class: {retrieval_super_class_id}')

    plt.subplots_adjust(top=1.4, bottom=0.01)
    plt.savefig(f'../retrieval_plots/{title}_retrieval.png')


def evaluate(model, annoy_index):
    acc_super_class_id = 0
    acc_class_id = 0
    map_super_class_id = 0
    map_class_id = 0
    for i in tqdm(range(len(X_val))):
        img = load_img_from_path(X_val.iloc[i]['path'])
        class_id = X_val.iloc[i]['class_id']
        super_class_id = X_val.iloc[i]['super_class_id']

        # get retrieval images
        feature_v = get_vector_for_img(img, model)
        neighbours = annoy_index.get_nns_by_vector(feature_v, 5)  # get top 5 closest

        curr_acc_class_id = 0
        curr_acc_superclass_id = 0
        same_class_id = 0
        same_super_class_idt = 0
        for indx, neighbour in enumerate(neighbours):
            retrieval_class_id = X_train.iloc[neighbour]['class_id']
            retrieval_super_class_id = X_train.iloc[neighbour]['super_class_id']

            if retrieval_super_class_id == super_class_id:
                same_super_class_idt += 1
                curr_acc_superclass_id += same_super_class_idt / (indx + 1)
            if retrieval_class_id == class_id:
                same_class_id += 1
                curr_acc_class_id += same_class_id / (indx + 1)

        acc_super_class_id += same_super_class_idt
        acc_class_id += same_class_id
        map_super_class_id += curr_acc_superclass_id / len(neighbours)
        map_class_id += same_class_id / len(neighbours)

    acc_super_class_id = acc_super_class_id / (len(neighbours) * len(X_val))
    acc_class_id = acc_class_id / (len(neighbours) * len(X_val))
    map_superclass_id = map_super_class_id / len(X_val)
    map_class_id = map_class_id / len(X_val)

    print(f'Top 5 accuracy for Super Class: {acc_super_class_id}')
    print(f'Top 5 accuracy for Class: {acc_class_id}')
    print(f'Map for Super Class: {map_superclass_id}')
    print(f'Map for Class: {map_class_id}')
