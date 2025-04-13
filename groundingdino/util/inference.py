from typing import Tuple, List

import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image
from torchvision.ops import box_convert
import bisect

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.misc import clean_state_dict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import get_phrases_from_posmap
from openai import OpenAI
from typing import Optional

import uuid
import base64
from GroundingDINO.config import settings

# ----------------------------------------------------------------------------------------------------------------------
# OLD API
# ----------------------------------------------------------------------------------------------------------------------


# def preprocess_caption(caption: str) -> str:
#     result = caption.lower().strip()
#     if result.endswith("."):
#         return result
#     return result + "."

client = OpenAI(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_API_BASE_URL)

def preprocess_caption(
    caption: str,
    language: Optional[str] = "en",  # 目标语言（默认为英语）
    extract_keywords: bool = True,    # 是否提取关键词
    # openai_api_key: str = "YOUR_API_KEY"
) -> str:
    """
    使用大模型增强的文本预处理：
    1. 翻译为目标语言
    2. 提取关键信息
    3. 标准化文本格式
    """
    # 初始化 OpenAI 客户端
    # openai.api_key = openai_api_key

    # 构造大模型指令
    system_prompt = f"""
    You are a text preprocessing assistant. Perform the following steps:
    1. Translate the input to {language} if not already in {language}.
    2. { "Extract key noun phrases, remove adjectives/adverbs if possible." if extract_keywords else "" }
    3. Format as lowercase without ending punctuation.
    4. Correct any spelling/grammar errors.
    Output ONLY the final result.
    """
    
    # # 调用大模型
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": caption}
    #     ],
    #     temperature=0.1  # 低随机性确保稳定性
    # )

    response = client.chat.completions.create(
        model=settings.OPENAI_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": caption},
        ],
        temperature=0.1,  # 低随机性确保稳定性
        stream=False
    )
    
    processed_text = response.choices[0].message.content.strip()
    
    # 确保以句号结尾（适配原检测模型）
    if not processed_text.endswith("."):
        processed_text += "."
    
    return processed_text.lower()


def load_model(model_config_path: str, model_checkpoint_path: str, device: str = "cuda"):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed


def predict(
        model,
        image: torch.Tensor,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        device: str = "cuda",
        remove_combined: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    caption = preprocess_caption(caption=caption)

    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)

    mask = prediction_logits.max(dim=1)[0] > box_threshold
    logits = prediction_logits[mask]  # logits.shape = (n, 256)
    boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    
    if remove_combined:
        sep_idx = [i for i in range(len(tokenized['input_ids'])) if tokenized['input_ids'][i] in [101, 102, 1012]]
        
        phrases = []
        for logit in logits:
            max_idx = logit.argmax()
            insert_idx = bisect.bisect_left(sep_idx, max_idx)
            right_idx = sep_idx[insert_idx]
            left_idx = sep_idx[insert_idx - 1]
            phrases.append(get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer, left_idx, right_idx).replace('.', ''))
    else:
        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
            for logit
            in logits
        ]

    return boxes, logits.max(dim=1)[0], phrases


# def annotate(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str]) -> np.ndarray:
#     """    
#     This function annotates an image with bounding boxes and labels.

#     Parameters:
#     image_source (np.ndarray): The source image to be annotated.
#     boxes (torch.Tensor): A tensor containing bounding box coordinates.
#     logits (torch.Tensor): A tensor containing confidence scores for each bounding box.
#     phrases (List[str]): A list of labels for each bounding box.

#     Returns:
#     np.ndarray: The annotated image.
#     """
#     h, w, _ = image_source.shape
#     boxes = boxes * torch.Tensor([w, h, w, h])
#     xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
#     detections = sv.Detections(xyxy=xyxy)

#     labels = [
#         f"{phrase} {logit:.2f}"
#         for phrase, logit
#         in zip(phrases, logits)
#     ]

#     bbox_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
#     label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
#     annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
#     annotated_frame = bbox_annotator.annotate(scene=annotated_frame, detections=detections)
#     annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
#     return annotated_frame

# 转化为label studio格式
def annotate(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str]) -> np.ndarray:
    """    
    This function annotates an image with bounding boxes and labels.

    Parameters:
    image_source (np.ndarray): The source image to be annotated.
    boxes (torch.Tensor): A tensor containing bounding box coordinates.
    logits (torch.Tensor): A tensor containing confidence scores for each bounding box.
    phrases (List[str]): A list of labels for each bounding box.

    Returns:
    np.ndarray: The annotated image.
    """
    h, w, _ = image_source.shape
    # 转换坐标到绝对像素值
    if boxes.numel() > 0:
        pixel_boxes = boxes * torch.Tensor([w, h, w, h]).to(boxes.device)
        xyxy = box_convert(boxes=pixel_boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
    else:
        xyxy = np.empty((0, 4))

    # 构建Label Studio标注结构
    annotations = []
    for bbox, score, phrase in zip(xyxy, logits, phrases):
        x1, y1, x2, y2 = bbox
        
        # 转换为百分比坐标
        x_percent = x1 / w * 100
        y_percent = y1 / h * 100
        width_percent = (x2 - x1) / w * 100
        height_percent = (y2 - y1) / h * 100

        annotations.append({
            "id": str(uuid.uuid4()),
            "type": "rectanglelabels",
            "value": {
                "x": x_percent,
                "y": y_percent,
                "width": width_percent,
                "height": height_percent,
                "rotation": 0,
                "rectanglelabels": [phrase]
            },
            "score": float(score),
            "from_name": "label",
            "to_name": "image"
        })

    # 构建完整任务结构
    task = {
        "data": {
            "image": f"data:image/jpeg;base64,{image_to_base64(image_source)}"
        },
        "predictions": [{
            "model_version": "grounding-dino",
            "score": float(logits.mean().item()) if len(logits) > 0 else 0.0,
            "result": annotations
        }]
    }
    
    return task

def image_to_base64(image_array: np.ndarray) -> str:
    """将OpenCV图像转为Base64字符串"""
    _, buffer = cv2.imencode('.jpg', image_array)
    return base64.b64encode(buffer).decode('utf-8')

# ----------------------------------------------------------------------------------------------------------------------
# NEW API
# ----------------------------------------------------------------------------------------------------------------------


class Model:

    def __init__(
        self,
        model_config_path: str,
        model_checkpoint_path: str,
        device: str = "cuda"
    ):
        self.model = load_model(
            model_config_path=model_config_path,
            model_checkpoint_path=model_checkpoint_path,
            device=device
        ).to(device)
        self.device = device

    def predict_with_caption(
        self,
        image: np.ndarray,
        caption: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25
    ) -> Tuple[sv.Detections, List[str]]:
        """
        import cv2

        image = cv2.imread(IMAGE_PATH)

        model = Model(model_config_path=CONFIG_PATH, model_checkpoint_path=WEIGHTS_PATH)
        detections, labels = model.predict_with_caption(
            image=image,
            caption=caption,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        import supervision as sv

        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(scene=image, detections=detections, labels=labels)
        """
        processed_image = Model.preprocess_image(image_bgr=image).to(self.device)
        boxes, logits, phrases = predict(
            model=self.model,
            image=processed_image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold, 
            device=self.device)
        source_h, source_w, _ = image.shape
        detections = Model.post_process_result(
            source_h=source_h,
            source_w=source_w,
            boxes=boxes,
            logits=logits)
        return detections, phrases

    def predict_with_classes(
        self,
        image: np.ndarray,
        classes: List[str],
        box_threshold: float,
        text_threshold: float
    ) -> sv.Detections:
        """
        import cv2

        image = cv2.imread(IMAGE_PATH)

        model = Model(model_config_path=CONFIG_PATH, model_checkpoint_path=WEIGHTS_PATH)
        detections = model.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )


        import supervision as sv

        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(scene=image, detections=detections)
        """
        caption = ". ".join(classes)
        processed_image = Model.preprocess_image(image_bgr=image).to(self.device)
        boxes, logits, phrases = predict(
            model=self.model,
            image=processed_image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device)
        source_h, source_w, _ = image.shape
        detections = Model.post_process_result(
            source_h=source_h,
            source_w=source_w,
            boxes=boxes,
            logits=logits)
        class_id = Model.phrases2classes(phrases=phrases, classes=classes)
        detections.class_id = class_id
        return detections

    @staticmethod
    def preprocess_image(image_bgr: np.ndarray) -> torch.Tensor:
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_pillow = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        image_transformed, _ = transform(image_pillow, None)
        return image_transformed

    @staticmethod
    def post_process_result(
            source_h: int,
            source_w: int,
            boxes: torch.Tensor,
            logits: torch.Tensor
    ) -> sv.Detections:
        boxes = boxes * torch.Tensor([source_w, source_h, source_w, source_h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        confidence = logits.numpy()
        return sv.Detections(xyxy=xyxy, confidence=confidence)

    @staticmethod
    def phrases2classes(phrases: List[str], classes: List[str]) -> np.ndarray:
        class_ids = []
        for phrase in phrases:
            for class_ in classes:
                if class_ in phrase:
                    class_ids.append(classes.index(class_))
                    break
            else:
                class_ids.append(None)
        return np.array(class_ids)
