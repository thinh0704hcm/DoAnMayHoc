import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from PIL import Image
import torch.optim as optim

# Kiểm tra xem CUDA có khả dụng không và thiết lập thiết bị sử dụng
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {device}")

# Định nghĩa thư mục gốc chứa dữ liệu đã giải nén
data_root = "/content/sorted_data"

# Lớp Dataset tùy chỉnh
class DigitDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Tham số:
            root_dir (string): Thư mục chứa tất cả các thư mục số.
            transform (callable, optional): Phép biến đổi sẽ được áp dụng lên mỗi mẫu (nếu có).
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        self.labels = []

        # Duyệt qua thư mục để tìm file ảnh và nhãn tương ứng
        for label_folder in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label_folder)
            if os.path.isdir(label_path):
                try:
                    label = int(label_folder) # Giả định tên thư mục là nhãn số (0-9)
                    for image_file in os.listdir(label_path):
                        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')): # Thêm định dạng ảnh nếu cần và xử lý phân biệt hoa thường
                            # Bỏ qua file ẩn
                            if not image_file.startswith('.'):
                                self.image_files.append(os.path.join(label_path, image_file))
                                self.labels.append(label)
                except ValueError:
                    print(f"Bỏ qua thư mục không phải số nguyên: {label_folder}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB') # Mở ảnh và chuyển sang RGB
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def load_and_split_data(data_root, train_transforms, test_transforms, train_split=0.8, batch_size=64, test_batch_size=1000):
    """
    Tải dữ liệu, chia thành tập huấn luyện và kiểm tra, và tạo DataLoader.

    Tham số:
        data_root (str): Thư mục chứa tất cả các thư mục số.
        train_transforms (callable): Các phép biến đổi cho tập huấn luyện.
        test_transforms (callable): Các phép biến đổi cho tập kiểm tra.
        train_split (float): Tỉ lệ dữ liệu dùng cho huấn luyện.
        batch_size (int): Kích thước batch cho DataLoader huấn luyện.
        test_batch_size (int): Kích thước batch cho DataLoader kiểm tra.

    Trả về:
        tuple: train_loader và test_loader.
    """
    try:
        full_dataset = DigitDataset(root_dir=data_root, transform=None)

        train_size = int(train_split * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_indices, test_indices = random_split(range(len(full_dataset)), [train_size, test_size])

        train_dataset = Subset(full_dataset, train_indices)
        test_dataset = Subset(full_dataset, test_indices)

        # Áp dụng phép biến đổi cho từng tập
        train_dataset.dataset.transform = train_transforms
        test_dataset.dataset.transform = test_transforms

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

        print("Đã tải dữ liệu, chia tập và tạo DataLoader.")
        print(f"Số lượng mẫu huấn luyện: {len(train_dataset)}")
        print(f"Số lượng mẫu kiểm tra: {len(test_dataset)}")

        return train_loader, test_loader

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy thư mục '{data_root}'. Vui lòng kiểm tra đường dẫn dữ liệu đã giải nén.")
        return None, None
    except Exception as e:
        print(f"Đã xảy ra lỗi khi tải dữ liệu: {e}")
        return None, None

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
        model.fc = nn.Linear(num_ftrs, num_classes)
        print(f"Đã sửa lớp cuối cho {num_classes} lớp.")

        model = model.to(device)
        print("Đã chuyển mô hình sang thiết bị.")

        return model
    except Exception as e:
        print(f"Đã xảy ra lỗi khi xây dựng mô hình: {e}")
        return None

def evaluate_model(model, dataloader, criterion, device):
    """
    Đánh giá hiệu năng của mô hình trên tập dữ liệu cho trước.

    Tham số:
        model (torch.nn.Module): Mô hình đã huấn luyện.
        dataloader (torch.utils.data.DataLoader): DataLoader cho tập đánh giá.
        criterion (torch.nn.Module): Hàm mất mát.
        device (torch.device): Thiết bị đánh giá.

    Trả về:
        tuple: Mất mát trung bình và độ chính xác.
    """
    model.eval()  # Đặt mô hình ở chế độ đánh giá
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  # Tắt tính toán gradient
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    average_loss = running_loss / total_samples
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy

def train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, num_epochs, device, save_path="handwritten_digit_model_weights.pth"):
    """
    Huấn luyện mô hình và đánh giá sau khi huấn luyện.

    Tham số:
        model (torch.nn.Module): Mô hình cần huấn luyện.
        train_loader (torch.utils.data.DataLoader): DataLoader cho tập huấn luyện.
        test_loader (torch.utils.data.DataLoader): DataLoader cho tập kiểm tra.
        optimizer (torch.optim.Optimizer): Bộ tối ưu.
        criterion (torch.nn.Module): Hàm mất mát.
        num_epochs (int): Số epoch huấn luyện.
        device (torch.device): Thiết bị huấn luyện (cuda hoặc cpu).
        save_path (str): Đường dẫn lưu trọng số mô hình.
    """
    model.train() # Đặt mô hình ở chế độ huấn luyện
    print("Bắt đầu huấn luyện...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            # Đưa dữ liệu lên thiết bị
            inputs, labels = inputs.to(device), labels.to(device)

            # Đặt lại gradient về 0
            optimizer.zero_grad()

            # Lan truyền tiến
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Lan truyền ngược và tối ưu
            loss.backward()
            optimizer.step()

            # Hiển thị thống kê
            running_loss += loss.item()
            if (i + 1) % 100 == 0: # In ra mỗi 100 batch
                print(f'Epoch [{epoch+1}/{num_epochs}], Bước [{i+1}/{len(train_loader)}], Mất mát: {loss.item():.4f}')

        print(f'Epoch [{epoch+1}/{num_epochs}] hoàn thành. Mất mát trung bình: {running_loss / len(train_loader):.4f}')

    try:
        # Lưu trọng số mô hình
        torch.save(model.state_dict(), save_path)
        print(f"Đã lưu trọng số mô hình tại {save_path}")
    except Exception as e:
        print(f"Đã xảy ra lỗi khi lưu mô hình: {e}")

    print("Đã hoàn thành huấn luyện.")
    print("Lớp cuối của mô hình:")
    print(model.fc)

    print("Đang đánh giá mô hình trên tập kiểm tra...")
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
    print(f"Mất mát kiểm tra: {test_loss:.4f}")
    print(f"Độ chính xác kiểm tra: {test_accuracy:.4f}")

# --- Thực thi chính ---

# Định nghĩa các phép biến đổi
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transforms = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Đường dẫn lưu trọng số mô hình
model_save_path = "handwritten_digit_model_weights.pth"

# Tải và chia dữ liệu
train_loader, test_loader = load_and_split_data(data_root, train_transforms, test_transforms)

if train_loader is not None and test_loader is not None:
    # Xây dựng mô hình
    num_classes = 10
    model = build_model(num_classes, device)

    if model is not None:
        # Thiết lập hàm mất mát và bộ tối ưu
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Kiểm tra nếu đã có trọng số mô hình lưu
        if os.path.exists(model_save_path):
            try:
                model.load_state_dict(torch.load(model_save_path))
                print(f"Đã tải trọng số mô hình từ {model_save_path}")
            except Exception as e:
                print(f"Lỗi khi tải trọng số mô hình: {e}")

        # Huấn luyện và đánh giá mô hình
        num_epochs = 12 # Số epoch
        train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, num_epochs, device, save_path=model_save_path)
