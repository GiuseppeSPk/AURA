
import json
import base64
import os

notebook_path = r"C:\Users\spicc\Downloads\aura-v10-kaggle.ipynb"
output_dir = r"c:\Users\spicc\Desktop\Multimodal\reports\graphs"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

img_count = 0
for cell in nb['cells']:
    if 'outputs' in cell:
        for output in cell['outputs']:
            if 'data' in output and 'image/png' in output['data']:
                img_data = output['data']['image/png']
                img_count += 1
                img_name = f"graph_{img_count}.png"
                with open(os.path.join(output_dir, img_name), 'wb') as f_img:
                    f_img.write(base64.b64decode(img_data))

print(f"Extracted {img_count} images to {output_dir}")
