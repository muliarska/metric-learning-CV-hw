from tqdm import tqdm
import torch
import torchvision
import numpy as np
from processing.data_processing import split_data
from processing.utils import load_img_from_path
from processing.utils import build_index, visualize_retrieval, evaluate
from pytorch_metric_learning.losses import ContrastiveLoss


X_train, _ = split_data()


# Training function.
def train_step(model, optimizer, loss_optimizer, device, data_dir):
    print("Training ResNet with Contrastive Loss")
    model.train()
    train_running_loss = 0.0
    counter = 0
    for i in tqdm(range(len(X_train))):
        img = load_img_from_path(data_dir + X_train.iloc[i]['path'])
        super_class_id = X_train.iloc[i]['super_class_id']
        labels = torch.from_numpy(np.array([int(el) for el in super_class_id]))

        optimizer.zero_grad()
        loss_optimizer.zero_grad()

        outputs = model(img)
        # Applying Contrastive loss
        loss = loss_optimizer(outputs, labels)
        train_running_loss += loss.item()

        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        # Backpropagation
        loss.backward()
        # Update the weights.
        optimizer.step()
        loss_optimizer.step()

    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    print("Epoch loss: ", round(epoch_loss, 3))


def resnet18_contrastive_loss_training():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    data_dir = '../../../../Downloads/Stanford_Online_Products/'

    X_train, _ = split_data(data_dir=data_dir)

    resnet18 = torchvision.models.resnet18(pretrained=True)
    resnet18_contrastive_loss = resnet18.to(device)

    optimizer = torch.optim.SGD(resnet18_contrastive_loss.parameters(), lr=0.001, momentum=0.9)

    # Contrastive loss
    contrastive_loss = ContrastiveLoss().to(device)

    epochs = 25
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch + 1} of {epochs}")
        train_step(resnet18_contrastive_loss, optimizer, contrastive_loss, device, data_dir)

    return resnet18_contrastive_loss


if __name__ == '__main__':
    title = "resnet18_contrastive_loss"

    print("STEP 1: Training the fine tuned ResNet with Contrastive Loss")
    resnet18_contrastive_loss = resnet18_contrastive_loss_training()

    print("\nSTEP 2: Building an index")
    resnet_index = build_index(resnet18_contrastive_loss, title)

    print("\nSTEP 3: Evaluating")
    evaluate(resnet18_contrastive_loss, resnet_index)

    print("\nSTEP 4: Performing a retrieval on resnet18 with Contrastive Loss")
    visualize_retrieval(resnet18_contrastive_loss, resnet_index, title)