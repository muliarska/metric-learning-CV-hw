import torchvision
from processing.utils import build_index, visualize_retrieval, evaluate


print("Loading resnet18")
resnet18 = torchvision.models.resnet18(pretrained=True)
resnet18.eval()

print("Building an index for resnet18")
resnet_index = build_index(resnet18, "resnet18")
resnet_index.get_nns_by_item(0, 5)

print("Evaluating resnet18")
evaluate(resnet18, resnet_index)

print("Performing a retrieval on resnet18")
visualize_retrieval(resnet18, resnet_index, "resnet18")
