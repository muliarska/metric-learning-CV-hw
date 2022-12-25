from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from processing.dataloader import MyDataset


# Define the model for the pretext colorization task
class DenoisingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = torch.nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv6 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv7 = torch.nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv8 = torch.nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv9 = torch.nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv10 = torch.nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv11 = torch.nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.relu(self.conv4(x))
        x = torch.nn.functional.relu(self.conv5(x))
        x = torch.nn.functional.relu(self.conv6(x))
        x = torch.nn.functional.relu(self.conv7(x))
        x = torch.nn.functional.relu(self.conv8(x))
        x = torch.nn.functional.relu(self.conv9(x))
        x = torch.nn.functional.relu(self.conv10(x))
        x = self.conv11(x)
        return x


# Define the loss function and optimizer for the pretext colorization task
denoised_model = DenoisingModel()
denoised_loss_fn = torch.nn.MSELoss()
denoised_optimizer = torch.optim.Adam(denoised_model.parameters())


path = "../../../../Downloads/Stanford_Online_Products/"
image_size = 250

train_dataloader = DataLoader(MyDataset(path, image_size, "train"), batch_size=32, shuffle=True, num_workers=4)

epochs = 25
# Define the training loop for the pretext colorization task
for epoch in tqdm(range(epochs)):
    print(f"[INFOt]: Epoch {epoch + 1} of {epochs}")
    print("Training using Self Supervised")
    for batch in train_dataloader:
        img, noised = batch
        # Predict the missing color channels
        outputs = denoised_model(noised)

        # Compute the loss and gradient
        loss = denoised_loss_fn(outputs, img)
        denoised_optimizer.zero_grad()
        loss.backward()

        # Update the model parameters
        denoised_optimizer.step()
