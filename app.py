from groundingdino.util.inference import load_model, load_image, predict, annotate
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from label_studio_sdk import Client
import random
import json
from typing import List
from uuid import uuid4
import os
import tempfile
import shutil
import zipfile
import tarfile
import rarfile
import py7zr
from config import settings

app = FastAPI()

model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")

# 连接到 Label Studio
LABEL_STUDIO_URL = settings.LABEL_STUDIO_URL
API_KEY = settings.LABEL_STUDIO_API_KEY
project_id = 1

client = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
project = client.get_project(project_id)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
ZIP_EXTENSIONS = {'.zip', '.tar', '.tar.gz', '.7z', '.rar'}

def get_extension(filename: str) -> str:
    """提取扩展名，统一为小写"""
    return os.path.splitext(filename.lower())[1]

def extract_archive(file_path, dest_folder):
    ext = get_extension(file_path)
    print(ext)
    if ext == '.7z':
        with py7zr.SevenZipFile(file_path, 'r') as archive:
            archive.extractall(dest_folder)
    elif ext == '.rar':
        with rarfile.RarFile(file_path, 'r') as rf:
            rf.extractall(dest_folder)
    elif ext.endswith(".zip"):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(dest_folder)

    elif ext.endswith(".tar") or ext.endswith(".tar.gz") or ext.endswith(".tgz") or ext.endswith(".tar.bz2"):
        with tarfile.open(file_path, 'r:*') as tar_ref:  # 自动识别压缩格式
            tar_ref.extractall(dest_folder)
    else:
        raise ValueError("Unsupported archive type")

def is_image_file(filename):
    return get_extension(filename) in IMAGE_EXTENSIONS

def collect_images(folder_path: str) -> List[str]:
    image_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if is_image_file(file):
                image_paths.append(os.path.join(root, file))
    return image_paths

@app.post("/annotate/")
async def annotate_image(
    prompt: str,
    file: UploadFile = File(...),
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    
):
    filename = file.filename
    extension = get_extension(filename)
    print(extension)

    if extension in IMAGE_EXTENSIONS:
        print('111111111')
        try:
            # 保存上传的图片到临时文件
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name

            # 读取图像
            image_source, image_tensor = load_image(tmp_path)

            # 执行模型预测
            boxes, logits, phrases = predict(
                model=model,
                image=image_tensor,
                caption=prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device='cpu'
            )

            # 构建Label Studio格式任务
            task = annotate(
                image_source=image_source,
                boxes=boxes,
                logits=logits,
                phrases=phrases
            )

            all_phrases = list(dict.fromkeys(phrases))
            print(f"提取到的标签: {all_phrases}")

            xml_config = generate_label_config(all_phrases)
            print(f"生成的标签配置: {xml_config}")
            project.set_params(label_config=xml_config)
            print(f"已更新项目 {project_id} 的标签配置")
            # 导入任务
            project.import_tasks([task])
            # 删除临时文件
            os.remove(tmp_path)

            return JSONResponse(content=task, status_code=200)

        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=500)

    elif extension in ZIP_EXTENSIONS or filename.lower().endswith('.tar.gz'):
        print('11111111133333333333')
        temp_dir = tempfile.mkdtemp()
        print(temp_dir)
        file_path = os.path.join(temp_dir, file.filename)
        print('111111111222222222')
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 解压缩
        extract_folder = os.path.join(temp_dir, "extracted")
        os.makedirs(extract_folder, exist_ok=True)
        print('111111111')
        extract_archive(file_path, extract_folder)

        # 收集图片
        image_paths = collect_images(extract_folder)
        if not image_paths:
            return JSONResponse(content={"error": "No images found in archive."}, status_code=400)
        for img_path in image_paths:
            try:
                # 读取图像
                image_source, image_tensor = load_image(img_path)

                # 执行模型预测
                boxes, logits, phrases = predict(
                    model=model,
                    image=image_tensor,
                    caption=prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    device='cpu'
                )

                # 构建Label Studio格式任务
                task = annotate(
                    image_source=image_source,
                    boxes=boxes,
                    logits=logits,
                    phrases=phrases
                )

                all_phrases = list(dict.fromkeys(phrases))
                print(f"提取到的标签: {all_phrases}")

                xml_config = generate_label_config(all_phrases)
                print(f"生成的标签配置: {xml_config}")
                project.set_params(label_config=xml_config)
                print(f"已更新项目 {project_id} 的标签配置")
                # 导入任务
                project.import_tasks([task])

                return JSONResponse(content=task, status_code=200)

            except Exception as e:
                return JSONResponse(content={"error": str(e)}, status_code=500)
    
    else:
        raise ValueError("Unsupported archive type")

def generate_label_config(labels: List[str]) -> str:
    """生成动态标签配置XML"""
    label_elements = "\n".join(
        f'<Label value="{escape_xml(label)}" background="#{generate_color(label)}"/>'
        for label in labels
    )
    
    return f'''<View>
  <Image name="image" value="$image" zoom="true"/>
  <RectangleLabels name="label" toName="image">
    {label_elements}
  </RectangleLabels>
</View>'''

def escape_xml(text: str) -> str:
    """转义XML特殊字符"""
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')

def generate_color(seed: str, min_brightness: int = 100) -> str:
    """基于标签名称生成一致性颜色"""
    random.seed(hash(seed))
    r = random.randint(min_brightness, 255)
    g = random.randint(min_brightness, 255)
    b = random.randint(min_brightness, 255)
    return f"{r:02X}{g:02X}{b:02X}"

import uvicorn
if __name__ == "__main__":
    uvicorn.run('app:app', reload=True, workers=1, host='0.0.0.0', port=8888)