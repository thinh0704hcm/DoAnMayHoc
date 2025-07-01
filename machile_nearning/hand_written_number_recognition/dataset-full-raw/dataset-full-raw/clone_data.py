import os
import subprocess
import glob

data_dir = 'handwritten_digit_classification'
os.makedirs(data_dir, exist_ok=True)

github_list = '''
https://github.com/DHPh/CS114_hand_written_digit/
https://github.com/Toan02Ky-UIT/CS114
https://github.com/23520276/Hand-written-digit-classification/
https://github.com/anngyn/CS114-Hand-Written-Digit
'''

# 1. Clone các repository từ danh sách
print("Cloning repositories...")
for url in github_list.strip().split():
    try:
        # Extract repository name
        repo_name = url.rstrip('/').split('/')[-1].replace('.git', '')
        dest_path = os.path.join(data_dir, repo_name)
        
        if not os.path.exists(dest_path):
            print(f"Cloning {url}")
            subprocess.run(['git', 'clone', '--depth', '1', url, dest_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone {url}: {str(e)}")

# 2. Đếm số lượng ảnh thu thập được
print("\nCollecting image paths...")
a = glob.glob(f'{data_dir}/*/hand_written_digit/??52????')
image_lists = []
for folder in a:
    for num in range(10):
        t = glob.glob(os.path.join(folder, f'{num}_*'))
        image_lists += t

print(f"Found {len(image_lists)} images")