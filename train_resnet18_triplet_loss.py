from tqdm import tqdm
import torch
import torchvision
from processing.data_processing import split_data
from processing.utils import build_index, visualize_retrieval, evaluate
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.miners import TripletMarginMiner
from pytorch_metric_learning.losses import TripletMarginLoss
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


# Training function.
def train_step(model, optimizer, triplet_margin_miner, triplet_margin_loss, device, data_dir):
    print("Training ResNet with Triplet Loss")
    model.train()
    train_running_loss = 0.0
    counter = 0
    for batch in tqdm(train_dataloader):
        img, _, super_class_id = batch
        labels = super_class_id

        optimizer.zero_grad()

        outputs = model(img)
        # Applying triplet loss
        loss = triplet_margin_loss(outputs, labels, triplet_margin_miner(outputs, labels))

        train_running_loss += loss.item()

        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        # Backpropagation
        loss.backward()
        # Update the weights.
        optimizer.step()

    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    print("Epoch loss: ", round(epoch_loss, 3))


def resnet18_triplet_loss_training():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    data_dir = '../../../../Downloads/Stanford_Online_Products/'

    X_train, _ = split_data(data_dir=data_dir)

    resnet18 = torchvision.models.resnet18(pretrained=True)
    resnet18_triplet_loss = resnet18.to(device)

    optimizer = torch.optim.SGD(resnet18_triplet_loss.parameters(), lr=0.001, momentum=0.9)

    # Triplet loss
    triplet_margin_miner = TripletMarginMiner(
        margin=0.2, distance=CosineSimilarity(), type_of_triplets="semihard"
    )
    triplet_margin_loss = TripletMarginLoss(
        margin=0.2, distance=CosineSimilarity(), reducer=ThresholdReducer(low=0)
    ).to(device)

    epochs = 25
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch + 1} of {epochs}")
        train_step(resnet18_triplet_loss, optimizer, triplet_margin_miner, triplet_margin_loss, device, data_dir)

    return resnet18_triplet_loss


if __name__ == '__main__':
    title = "resnet18_triplet_loss"

    print("STEP 1: Training the fine tuned ResNet with Triplet Loss")
    resnet18_triplet_loss = resnet18_triplet_loss_training()

    print("\nSTEP 2: Building an index")
    resnet_index = build_index(resnet18_triplet_loss, title)

    print("\nSTEP 3: Evaluating")
    evaluate(resnet18_triplet_loss, resnet_index)

    print("\nSTEP 4: Performing a retrieval on resnet18 with Triplet Loss")
    visualize_retrieval(resnet18_triplet_loss, resnet_index, title)