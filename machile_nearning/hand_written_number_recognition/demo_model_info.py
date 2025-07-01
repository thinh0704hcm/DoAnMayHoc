# Script demo để hiển thị thông tin lớp cuối cùng cho tất cả các mô hình
import torch
import torch.nn as nn
import torchvision.models as models

def print_separator(title):
    print("\n" + "="*60)
    print(f"{title}")
    print("="*60)

def count_model_layers(model):
    """Đếm tổng số lớp trong mô hình"""
    layer_count = 0
    for name, module in model.named_modules():
        # Đếm các lớp có ý nghĩa (không phải container)
        if len(list(module.children())) == 0:  # Chỉ các module lá
            if not isinstance(module, (torch.nn.Identity, torch.nn.Dropout)):
                layer_count += 1
    return layer_count

def print_model_info(model, model_name):
    """In thông tin về lớp cuối cùng của mô hình trước và sau khi chỉnh sửa"""
    print(f"\nMô hình: {model_name}")
    print(f"Loại: {type(model).__name__}")
    print(f"Tổng số tham số: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Tổng số lớp: {count_model_layers(model)}")
    
    print("\n" + "="*50)
    print("TRƯỚC KHI THAY ĐỔI (Mô hình gốc đã được huấn luyện trước)")
    print("="*50)
    
    # Handle different model types
    if hasattr(model, '_fc'):  # EfficientNet
        print(f"Tên lớp cuối: _fc")
        print(f"Loại lớp cuối: {type(model._fc).__name__}")
        print(f"Lớp cuối: {model._fc}")
        print(f"Số đặc trưng đầu vào: {model._fc.in_features}")
        print(f"Số đặc trưng đầu ra: {model._fc.out_features}")
        
        # Store original info
        original_in = model._fc.in_features
        original_out = model._fc.out_features
        original_layer = str(model._fc)
        
        # Modify the layer
        model._fc = nn.Linear(model._fc.in_features, 10)
        
        print("\n" + "="*50)
        print("SAU KHI THAY ĐỔI (Chỉnh sửa cho bài toán nhận diện chữ số)")
        print("="*50)
        print(f"Tên lớp cuối: _fc")
        print(f"Loại lớp cuối: {type(model._fc).__name__}")
        print(f"Lớp cuối: {model._fc}")
        print(f"Số đặc trưng đầu vào: {model._fc.in_features}")
        print(f"Số đặc trưng đầu ra: {model._fc.out_features}")
        print(f"Tổng số lớp: {count_model_layers(model)}")
        
        print("\n" + "SO SÁNH TRƯỚC VÀ SAU")
        print("-" * 40)
        print(f"TRƯỚC: Linear({original_in}, {original_out})")
        print(f"SAU:   Linear({model._fc.in_features}, {model._fc.out_features})")
        print(f"Thay đổi: {original_out} -> {model._fc.out_features} đặc trưng")
        
    elif hasattr(model, 'fc'):  # ResNet
        print(f"Tên lớp cuối: fc")
        print(f"Loại lớp cuối: {type(model.fc).__name__}")
        print(f"Lớp cuối: {model.fc}")
        print(f"Số đặc trưng đầu vào: {model.fc.in_features}")
        print(f"Số đặc trưng đầu ra: {model.fc.out_features}")
        
        # Store original info
        original_in = model.fc.in_features
        original_out = model.fc.out_features
        original_layer = str(model.fc)
        
        # Modify the layer
        model.fc = nn.Linear(model.fc.in_features, 10)
        
        print("\n" + "="*50)
        print("SAU KHI THAY ĐỔI (Chỉnh sửa cho bài toán nhận diện chữ số)")
        print("="*50)
        print(f"Tên lớp cuối: fc")
        print(f"Loại lớp cuối: {type(model.fc).__name__}")
        print(f"Lớp cuối: {model.fc}")
        print(f"Số đặc trưng đầu vào: {model.fc.in_features}")
        print(f"Số đặc trưng đầu ra: {model.fc.out_features}")
        print(f"Tổng số lớp: {count_model_layers(model)}")
        
        print("\n" + "SO SÁNH TRƯỚC VÀ SAU")
        print("-" * 40)
        print(f"TRƯỚC: Linear({original_in}, {original_out})")
        print(f"SAU:   Linear({model.fc.in_features}, {model.fc.out_features})")
        print(f"Thay đổi: {original_out} -> {model.fc.out_features} đặc trưng")

    elif hasattr(model, 'head'):  # Vision Transformer
        print(f"Tên lớp cuối: head")
        print(f"Loại lớp cuối: {type(model.head).__name__}")
        print(f"Head: {model.head}")
        if hasattr(model.head, 'in_features'):
            print(f"Số đặc trưng đầu vào: {model.head.in_features}")
            print(f"Số đặc trưng đầu ra: {model.head.out_features}")
            
            # Store original info
            original_in = model.head.in_features
            original_out = model.head.out_features
            original_layer = str(model.head)
            
            # Modify the layer
            model.head = nn.Linear(model.head.in_features, 10)
            
            print("\n" + "="*50)
            print("SAU KHI THAY ĐỔI (Chỉnh sửa cho bài toán nhận diện chữ số)")
            print("="*50)
            print(f"Tên lớp cuối: head")
            print(f"Loại lớp cuối: {type(model.head).__name__}")
            print(f"Head: {model.head}")
            print(f"Số đặc trưng đầu vào: {model.head.in_features}")
            print(f"Số đặc trưng đầu ra: {model.head.out_features}")
            print(f"Tổng số lớp: {count_model_layers(model)}")
            
            print("\n" + "SO SÁNH TRƯỚC VÀ SAU")
            print("-" * 40)
            print(f"TRƯỚC: Linear({original_in}, {original_out})")
            print(f"SAU:   Linear({model.head.in_features}, {model.head.out_features})")
            print(f"Thay đổi: {original_out} -> {model.head.out_features} đặc trưng")
        
    elif hasattr(model, 'classifier'):  # Other models
        print(f"Tên lớp cuối: classifier")
        print(f"Loại lớp cuối: {type(model.classifier).__name__}")
        print(f"Classifier: {model.classifier}")
        if hasattr(model.classifier, 'in_features'):
            print(f"Số đặc trưng đầu vào: {model.classifier.in_features}")
            print(f"Số đặc trưng đầu ra: {model.classifier.out_features}")
            
            # Store original info
            original_in = model.classifier.in_features
            original_out = model.classifier.out_features
            original_layer = str(model.classifier)
            
            # Modify the layer
            model.classifier = nn.Linear(model.classifier.in_features, 10)
            
            print("\n" + "="*50)
            print("SAU KHI THAY ĐỔI (Chỉnh sửa cho bài toán nhận diện chữ số)")
            print("="*50)
            print(f"Tên lớp cuối: classifier")
            print(f"Loại lớp cuối: {type(model.classifier).__name__}")
            print(f"Classifier: {model.classifier}")
            print(f"Số đặc trưng đầu vào: {model.classifier.in_features}")
            print(f"Số đặc trưng đầu ra: {model.classifier.out_features}")
            print(f"Tổng số lớp: {count_model_layers(model)}")
            
            print("\n" + "SO SÁNH TRƯỚC VÀ SAU")
            print("-" * 40)
            print(f"TRƯỚC: Linear({original_in}, {original_out})")
            print(f"SAU:   Linear({model.classifier.in_features}, {model.classifier.out_features})")
            print(f"Thay đổi: {original_out} -> {model.classifier.out_features} đặc trưng")

def main():
    print_separator("NHẬN DẠNG CHỮ SỐ VIẾT TAY - SO SÁNH LỚP CUỐI CÙNG")
    print("Demo này hiển thị thông tin lớp cuối TRƯỚC và SAU KHI THAY ĐỔI")
    print("cho tất cả các mô hình được sử dụng trong các script huấn luyện.")
    print("TRƯỚC THAY ĐỔI = Mô hình gốc đã được huấn luyện trước")
    print("SAU THAY ĐỔI = Chỉnh sửa cho phân loại chữ số 10 lớp")
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nThiết bị: {device}")
    
    models_to_test = []

    
    # Test Vision Transformer (ViT)
    print_separator("VISION TRANSFORMER (ViT)")
    try:
        import timm
        model = timm.create_model("vit_base_patch16_224")
        print_model_info(model, "Vision Transformer (ViT)")
        models_to_test.append(("Vision Transformer (ViT)", "Có sẵn"))
    except ImportError:
        print("Vision Transformer không có sẵn (pip install torchvision)")
        models_to_test.append(("Vision Transformer (ViT)", "Không có sẵn"))
    except Exception as e:
        print(f"Lỗi khi tải Vision Transformer: {e}")
        models_to_test.append(("Vision Transformer (ViT)", f"Lỗi: {str(e)[:50]}"))

    # Test EfficientNet-B4
    print_separator("EFFICIENTNET-B4")
    try:
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained("efficientnet-b4")
        print_model_info(model, "EfficientNet-B4")
        models_to_test.append(("EfficientNet-B4", "Có sẵn"))
    except ImportError:
        print("EfficientNet không có sẵn (pip install efficientnet_pytorch)")
        models_to_test.append(("EfficientNet-B4", "Không có sẵn"))
    except Exception as e:
        print(f"Lỗi khi tải EfficientNet-B4: {e}")
        models_to_test.append(("EfficientNet-B4", f"Lỗi: {str(e)[:50]}"))
    
    # Test EfficientNet-B7 (if available)
    print_separator("EFFICIENTNET-B7")
    try:
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained("efficientnet-b7")
        print_model_info(model, "EfficientNet-B7")
        models_to_test.append(("EfficientNet-B7", "Có sẵn"))
    except ImportError:
        print("EfficientNet không có sẵn (pip install efficientnet_pytorch)")
        models_to_test.append(("EfficientNet-B7", "Không có sẵn"))
    except Exception as e:
        print(f"Lỗi khi tải EfficientNet-B7: {e}")
        models_to_test.append(("EfficientNet-B7", f"Lỗi: {str(e)[:50]}"))

    # Test ResNet18
    print_separator("RESNET18")
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        print_model_info(model, "ResNet18")
        models_to_test.append(("ResNet18", "Có sẵn"))
    except Exception as e:
        print(f"Lỗi khi tải ResNet18: {e}")
        models_to_test.append(("ResNet18", f"Lỗi: {str(e)[:50]}"))
    
    # Test ResNet50
    print_separator("RESNET50")
    try:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        print_model_info(model, "ResNet50")
        models_to_test.append(("ResNet50", "Có sẵn"))
    except Exception as e:
        print(f"Lỗi khi tải ResNet50: {e}")
        models_to_test.append(("ResNet50", f"Lỗi: {str(e)[:50]}"))
    
    # Test ResNet101
    print_separator("RESNET101")
    try:
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        print_model_info(model, "ResNet101")
        models_to_test.append(("ResNet101", "Có sẵn"))
    except Exception as e:
        print(f"Lỗi khi tải ResNet101: {e}")
        models_to_test.append(("ResNet101", f"Lỗi: {str(e)[:50]}"))
    
    # Summary
    print_separator("TỔNG KẾT")
    print("Tình trạng có sẵn và trạng thái mô hình:")
    for model_name, status in models_to_test:
        print(f"  {model_name:<20} {status}")
    
    print("\nTất cả các mô hình được tải thành công đều hiển thị cả:")
    print("- TRƯỚC THAY ĐỔI: Kiến trúc được huấn luyện trước gốc")
    print("- SAU THAY ĐỔI: Chỉnh sửa cho phân loại chữ số viết tay 10 lớp")
    print("\nBạn có thể chạy các script train_local.py riêng lẻ để xem")
    print("thiết lập huấn luyện đầy đủ với khả năng tải dữ liệu.")

if __name__ == "__main__":
    main()
