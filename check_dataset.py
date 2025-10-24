import os
from PIL import Image

def check_dataset():
    print("🔍 Checking dataset...")
    
    folders = [
        'data/real_faces/train',
        'data/real_faces/val', 
        'data/fake_faces/train',
        'data/fake_faces/val'
    ]
    
    total_images = 0
    
    for folder in folders:
        if os.path.exists(folder):
            images = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"📁 {folder}: {len(images)} images")
            total_images += len(images)
            
            # Check first few images
            for img_file in images[:2]:
                try:
                    img_path = os.path.join(folder, img_file)
                    with Image.open(img_path) as img:
                        print(f"   ✅ {img_file} - Size: {img.size}, Format: {img.format}")
                except Exception as e:
                    print(f"   ❌ {img_file} - Error: {e}")
        else:
            print(f"❌ Folder missing: {folder}")
    
    print(f"\n📊 TOTAL IMAGES: {total_images}")
    print("✅ Dataset check complete!")

if __name__ == "__main__":
    check_dataset()