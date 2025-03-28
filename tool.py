import os
import shutil
import random
from tqdm import tqdm
from datasets.animal.translate import translate
idx = 0

# for label_ in tqdm(os.listdir('animal/raw-img')):
#     cnt = 0
#     label = translate[label_]
#     for img_name in os.listdir(f'animal/raw-img/{label_}'):
#         if cnt > 200: break
#         img_path = f'animal/raw-img/{label_}/{img_name}'
#         dest_path = f'animal_labeled/{label}/{img_name}'
#         if not os.path.exists(f'animal_labeled/{label}'): 
#             os.makedirs(f'animal_labeled/{label}')
#         shutil.copy(img_path, dest_path)
#         cnt += 1

# exit()

for dirpath, dirnames, files in os.walk('./animal/raw-img'):
    for filename in files:
        if random.randint(0, 100) == 1:
            if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
                path = os.path.join(dirpath, filename)
                # 复制到某目录
                dest_path = os.path.join('./animal_val', str(idx) + '.' +filename.split('.')[1])
                shutil.copy(path, dest_path)
                idx += 1
