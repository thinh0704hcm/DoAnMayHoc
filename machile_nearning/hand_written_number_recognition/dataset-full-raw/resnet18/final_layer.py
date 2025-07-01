import torch
from torchvision import models
from torch import nn

num_classes = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(num_classes, device):
    """
    Xây dựng mô hình transfer learning bằng cách tải mô hình pretrained và chỉnh sửa lớp cuối.

    Tham số:
        num_classes (int): Số lượng lớp đầu ra.
        device (torch.device): Thiết bị để tải mô hình lên.

    Trả về:
        torch.nn.Module: Mô hình đã cấu hình.
    """
    try:
        model = models.resnet18(pretrained=True)
        print("Đã tải mô hình ResNet18 pretrained.")

        num_ftrs = model.fc.in_features
        print(model.fc)
        model.fc = nn.Linear(num_ftrs, num_classes)
        print(f"Đã sửa lớp cuối cho {num_classes} lớp.")

        model = model.to(device)
        print("Đã chuyển mô hình sang thiết bị.")

        return model
    except Exception as e:
        print(f"Đã xảy ra lỗi khi xây dựng mô hình: {e}")
        return None

# Build model
model = build_model(num_classes, device)

# Load saved weights
model.load_state_dict(torch.load("272477.pth", map_location=device))

# Print final layer
print("Final layer of the model:", model.fc)

# Print the shape of the final layer
print("Shape of final layer weight:", model.fc.weight.shape)
print("Shape of final layer bias:", model.fc.bias.shape)

# Additional information about the final layer
print(f"Input features: {model.fc.in_features}")
print(f"Output features: {model.fc.out_features}")