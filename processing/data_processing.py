import pandas as pd
from sklearn.model_selection import train_test_split


def get_train_test_df(data_dir='../../../../../Downloads/Stanford_Online_Products/'):
    train_df = pd.read_csv(f'{data_dir}Ebay_train.txt', sep=" ", header=None)
    train_df.columns = ["image_id", "class_id", "super_class_id", "path"]
    train_df = train_df.iloc[1:, :]

    test_df = pd.read_csv(f'{data_dir}Ebay_test.txt', sep=" ", header=None)
    test_df.columns = ["image_id", "class_id", "super_class_id", "path"]
    test_df = test_df.iloc[1:, :]

    return train_df, test_df


def split_data(data_dir='../../../../../Downloads/Stanford_Online_Products/'):
    train_df, _ = get_train_test_df(data_dir)
    return train_test_split(train_df, test_size=0.33, random_state=42, stratify=train_df[['class_id']])


if __name__ == '__main__':
    train_df, test_df = get_train_test_df()
    X_train, X_val = split_data()
    print(len(X_train))
