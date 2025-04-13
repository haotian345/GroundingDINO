import base64
import requests
import os
import json
from label_studio_sdk import Client
from PIL import Image, ImageDraw, ImageFont
import io
from config import settings

LABEL_STUDIO_URL = settings.LABEL_STUDIO_URL
API_KEY = settings.LABEL_STUDIO_API_KEY

# 连接到 Label Studio
ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)  # 替换为你的地址和 token
project_id=1
project = ls.get_project(project_id)  # 替换为你的项目 ID

# 获取任务数据（包括图片）
tasks = project.get_tasks()

# 获取字体（如果需要）
font = ImageFont.load_default()

# 定义保存图片的文件夹
save_folder = "exported_images"
os.makedirs(save_folder, exist_ok=True)

# 遍历任务数据，提取图片并保存
for i, task in enumerate(tasks):
    image_data = task['data']['image']
    annotations = task['annotations']
    print(task)
    
    # 如果图片是 base64 编码格式
    if image_data.startswith('data:image'):
        base64_str = image_data.split(",")[1]
        img_bytes = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_bytes))
        
        with open(f"{save_folder}/image_{i}.jpg", 'wb') as img_file:
            img_file.write(img_bytes)
        print(f"保存图片: image_{i}.jpg")
    
        # 获取标注信息
        for annotation in annotations:
            for result in annotation['result']:
                if result['type'] == 'rectanglelabels':
                    # 获取边界框坐标和标签
                    x = result['value']['x']
                    y = result['value']['y']
                    width = result['value']['width']
                    height = result['value']['height']
                    label = result['value']['rectanglelabels'][0]

                    # 计算坐标
                    x1 = int(x * img.width / 100)
                    y1 = int(y * img.height / 100)
                    x2 = int((x + width) * img.width / 100)
                    y2 = int((y + height) * img.height / 100)

                    # 在图片上绘制矩形框和标签
                    draw = ImageDraw.Draw(img)
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=1)
                    draw.text((x1, y1), label, fill="red", font=font)

        # 保存带标注的图片
        img.save(f"{save_folder}/image_{i}_annotated.jpg")
        print(f"保存带标注的图片: image_{i}_annotated.jpg")
