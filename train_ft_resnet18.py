from tqdm import tqdm
import torch
import torchvision
import numpy as np
from processing.data_processing import split_data
from processing.utils import load_img_from_path


# Training function.
def train(model, optimizer, criterion, device, data_dir):
    print("Training ResNet fine tuned")
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i in tqdm(range(len(X_train))):
        img = load_img_from_path(data_dir + X_train.iloc[i]['path'])
        super_class_id = X_train.iloc[i]['super_class_id']
        labels = torch.from_numpy(np.array([int(el) for el in super_class_id]))

        optimizer.zero_grad()

        outputs = model(img)
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()

        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # Backpropagation
        loss.backward()
        # Update the weights.
        optimizer.step()

    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(X_train))
    print("Epoch loss: ", round(epoch_loss, 3))
    print("Epoch acc: ", round(epoch_acc, 3))
    return model


if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    data_dir = 'Stanford_Online_Products/'

    X_train, _ = split_data(data_dir=data_dir)

    resnet18 = torchvision.models.resnet18(pretrained=True)
    features_number = resnet18.fc.in_features
    # linear layer to add
    resnet18.fc = torch.nn.Linear(features_number, 12)
    resnet18_ft = resnet18.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(resnet18_ft.parameters(), lr=0.001, momentum=0.9)

    epochs = 25
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch + 1} of {epochs}")
        train(resnet18_ft, optimizer, criterion, device, data_dir)
