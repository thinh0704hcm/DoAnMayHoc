# Submission ID :         273738
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os

# Kiểm tra GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {device}")

# 1. Load và chuẩn bị dữ liệu với xử lý NaN
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    
    # Chuyển đổi cột điểm và xử lý NaN
    df['th_score'] = pd.to_numeric(df['th_score'], errors='coerce')
    
    # Tách dữ liệu có nhãn và không nhãn
    labeled = df.dropna(subset=['th_score']).copy()
    unlabeled = df[df['th_score'].isnull()].copy()
    
    # Kiểm tra và xử lý NaN trong features
    print(f"Số lượng NaN trong avg_score_scaled: {labeled['avg_score_scaled'].isna().sum()}")
    labeled = labeled.dropna(subset=['avg_score_scaled'])
    
    # Chuẩn bị dữ liệu huấn luyện
    X = labeled['avg_score_scaled'].values.astype(np.float32).reshape(-1, 1)
    y = labeled['th_score'].values.astype(np.float32).reshape(-1, 1)
    
    return X, y, labeled, unlabeled

try:
    X, y, labeled, unlabeled = load_and_prepare_data('../dataset-full-raw/student_simple_results.csv')
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file dữ liệu. Kiểm tra đường dẫn.")
    exit()

# 2. Chia dữ liệu train/val
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Kiểm tra NaN trong tập validation
print(f"NaN trong y_val: {np.isnan(y_val).sum()}")

# Chuyển đổi sang tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

# DataLoader
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 3. Định nghĩa mô hình
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Tăng learning rate
criterion = nn.MSELoss()

# 4. Huấn luyện với kiểm tra NaN
def evaluate(loader):
    model.eval()
    losses = []
    preds = []
    targets = []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb)
            
            # Kiểm tra NaN trong dự đoán
            if torch.isnan(out).any():
                print("Cảnh báo: Phát hiện NaN trong dự đoán")
                out = torch.nan_to_num(out, nan=0.0)
                
            loss = criterion(out, yb)
            losses.append(loss.item())
            preds.append(out.cpu().numpy())
            targets.append(yb.cpu().numpy())
    
    preds = np.vstack(preds)
    targets = np.vstack(targets)
    
    # Kiểm tra NaN trước khi tính toán metrics
    if np.isnan(preds).any() or np.isnan(targets).any():
        print("Cảnh báo: NaN trong dự đoán hoặc nhãn")
        preds = np.nan_to_num(preds, nan=0.0)
        targets = np.nan_to_num(targets, nan=0.0)
    
    mse = mean_squared_error(targets, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, preds)
    return np.mean(losses), mse, rmse, r2

def train_model(epochs=2000, patience=10):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    checkpoint_path = 'best_model.pth'
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            
            # Kiểm tra NaN trong loss
            loss = criterion(out, yb)
            if torch.isnan(loss):
                print("Cảnh báo: Loss bị NaN, bỏ qua batch này")
                continue
                
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        # Đánh giá validation
        val_loss, val_mse, val_rmse, val_r2 = evaluate(val_loader)
        
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train Loss {np.mean(train_losses):.4f} | Val Loss {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Dừng sớm do hiệu suất không cải thiện")
                break
    
    print("Huấn luyện hoàn tất!")
    model.load_state_dict(torch.load(checkpoint_path))

# 5. Huấn luyện và đánh giá
train_model()

# 6. Dự đoán cho sinh viên thiếu điểm
if not unlabeled.empty:
    # Lọc bỏ NaN trong features
    unlabeled = unlabeled.dropna(subset=['avg_score_scaled'])
    X_pred = torch.tensor(
        unlabeled['avg_score_scaled'].values.astype(np.float32).reshape(-1, 1)
    ).to(device)
    
    with torch.no_grad():
        pred_th = model(X_pred).cpu().numpy().flatten()
    
    result = pd.DataFrame({
        'student_id': unlabeled['student_id'],
        'predicted_th_score': pred_th
    })
    result.to_csv('student_predictions.csv', index=False)
    print(f"\nĐã dự đoán điểm TH cho {len(result)} sinh viên")
else:
    print("\nKhông có sinh viên nào cần dự đoán")

# 7. Đánh giá mô hình
print("\nĐánh giá mô hình cuối cùng:")
train_loss, train_mse, train_rmse, train_r2 = evaluate(train_loader)
val_loss, val_mse, val_rmse, val_r2 = evaluate(val_loader)

print(f"[Huấn luyện] MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
print(f"[Validation] MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")

# 8. Lưu tham số mô hình
print("\nTham số mô hình tốt nhất:")
for name, param in model.named_parameters():
    print(f"{name}: {param.data.cpu().numpy().flatten()}")
