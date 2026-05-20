import os
from PIL import Image

dir_path = r"c:\Users\vinay\PROJECTS\soil-intelligence-system\soil_ai_system\datasets\Soil-Classification-Dataset\Orignal-Dataset"

for root, dirs, files in os.walk(dir_path):
    for f in files:
        filepath = os.path.join(root, f)
        try:
            with Image.open(filepath) as img:
                rgb_im = img.convert('RGB')
                # Save as a standard JPEG, replacing the original file
                rgb_im.save(filepath, 'JPEG')
        except Exception as e:
            print(f"Removing invalid image: {filepath} ({e})")
            try:
                os.remove(filepath)
            except:
                pass

print("Finished standardizing images.")
