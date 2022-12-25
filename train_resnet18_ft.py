from tqdm import tqdm
import torch
import torchvision
from processing.data_processing import split_data
from processing.utils import build_index, visualize_retrieval, evaluate
from torch.utils.data import DataLoader
from processing.dataloader import MyDataset
import albumentations as albu


# X_train, _ = split_data()
path = "../../../../Downloads/Stanford_Online_Products/"
image_size = 250
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

train_dataloader = DataLoader(MyDataset(path, image_size, "train", albu), batch_size=32, shuffle=True, num_workers=4)
train_dataset_len = 39899

# Training function.
def train_step(model, optimizer, criterion, device, data_dir):
    print("Training ResNet fine tuned")
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for batch in tqdm(train_dataloader):
        img, _, super_class_id = batch
        labels = super_class_id

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
    epoch_acc = 100. * (train_running_correct / train_dataset_len)
    print("Epoch loss: ", round(epoch_loss, 3))
    print("Epoch acc: ", round(epoch_acc, 3))


def resnet18_ft_training():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    data_dir = '../../../../Downloads/Stanford_Online_Products/'

    X_train, _ = split_data(data_dir=data_dir)

    resnet18 = torchvision.models.resnet18(pretrained=True)
    features_number = resnet18.fc.in_features
    # linear layer to add
    resnet18.fc = torch.nn.Linear(features_number, 12)
    resnet18_ft = resnet18.to(device)

    # cross entropy used
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(resnet18_ft.parameters(), lr=0.001, momentum=0.9)

    epochs = 25
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch + 1} of {epochs}")
        train_step(resnet18_ft, optimizer, criterion, device, data_dir)
    return resnet18_ft


if __name__ == '__main__':
    title = "resnet18_ft"

    print("STEP 1: Training the fine tuned ResNet")
    resnet18_ft = resnet18_ft_training()

    print("\nSTEP 2: Building an index")
    resnet_index = build_index(resnet18_ft, title)

    print("\nSTEP 3: Evaluating")
    evaluate(resnet18_ft, resnet_index)

    print("\nSTEP 4: Performing a retrieval on resnet18 fine tuned")
    visualize_retrieval(resnet18_ft, resnet_index, title)
