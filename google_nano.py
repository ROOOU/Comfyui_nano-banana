import os
import io
import base64
import traceback
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

# 可选：批量模式支持 XLSX 需要 pandas 和 openpyxl
try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

# Google AI SDK
try:
    from google import genai
    from google.genai import types
    HAS_GOOGLE_AI = True
except Exception as e:
    HAS_GOOGLE_AI = False
    genai = None
    types = None


def _tensor_to_pils(image) -> List[Image.Image]:
    """
    将 ComfyUI 的 IMAGE(tensor[B,H,W,3], 浮点0-1) 转成 PIL 列表
    """
    if isinstance(image, dict) and "images" in image:
        image = image["images"]
    if not isinstance(image, torch.Tensor):
        raise TypeError("IMAGE 输入应为 torch.Tensor 或包含 'images' 键的 dict")
    if image.ndim == 3:
        image = image.unsqueeze(0)
    imgs = []
    arr = (image.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)  # [B,H,W,3]
    for i in range(arr.shape[0]):
        pil = Image.fromarray(arr[i], mode="RGB")
        imgs.append(pil)
    return imgs


def _ensure_pil_image(img) -> Image.Image:
    """
    确保输入是一个有效的 PIL Image 对象
    处理 google-genai 返回的 Image 类型
    """
    # 如果已经是 PIL Image，直接返回
    if isinstance(img, Image.Image):
        return img
    
    # 如果有 _pil_image 属性（google-genai 的 Image 类型）
    if hasattr(img, '_pil_image'):
        return img._pil_image
    
    # 如果有 data 属性，尝试从 bytes 创建
    if hasattr(img, 'data'):
        import io
        return Image.open(io.BytesIO(img.data))
    
    # 尝试直接转换
    try:
        return Image.fromarray(np.array(img))
    except Exception:
        raise TypeError(f"无法转换为 PIL Image: {type(img)}")


def _pils_to_tensor(pils: list) -> torch.Tensor:
    """
    将 PIL 列表转回 ComfyUI 的 IMAGE tensor[B,H,W,3], float32 0-1
    如果图片尺寸不一致，则分别处理每张图片，不强制统一尺寸
    """
    if not pils:
        # 返回一个空的占位张量，避免下游崩溃（B=0）
        return torch.zeros((0, 64, 64, 3), dtype=torch.float32)
    
    # 确保所有图片都是 PIL Image
    pils = [_ensure_pil_image(p) for p in pils]
    
    # 如果只有一张图片，直接处理
    if len(pils) == 1:
        pil = pils[0]
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        arr = np.array(pil, dtype=np.uint8)  # [H,W,3]
        tensor = torch.from_numpy(arr.astype(np.float32) / 255.0)  # [H,W,3]
        return tensor.unsqueeze(0)  # [1,H,W,3]
    
    # 检查所有图片是否具有相同尺寸
    first_size = (pils[0].width, pils[0].height)
    all_same_size = all((pil.width, pil.height) == first_size for pil in pils)
    
    if all_same_size:
        # 所有图片尺寸相同，可以直接堆叠
        np_imgs = []
        for pil in pils:
            if pil.mode != "RGB":
                pil = pil.convert("RGB")
            arr = np.array(pil, dtype=np.uint8)  # [H,W,3]
            np_imgs.append(arr)
        batch = np.stack(np_imgs, axis=0).astype(np.float32) / 255.0  # [B,H,W,3]
        return torch.from_numpy(batch)
    else:
        # 图片尺寸不同，只返回第一张图片，并在状态中说明
        # 这是ComfyUI的限制：IMAGE类型要求batch中所有图片尺寸相同
        pil = pils[0]
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        arr = np.array(pil, dtype=np.uint8)  # [H,W,3]
        tensor = torch.from_numpy(arr.astype(np.float32) / 255.0)  # [H,W,3]
        return tensor.unsqueeze(0)  # [1,H,W,3]


class GoogleNanoNode:
    """
    Google Nano (Flash) 节点 - 使用 Gemini 2.5 Flash Image 模型
    
    特点：
    - 快速生成
    - 支持最多 8 张参考图像
    - 适合高频率、低延迟任务
    
    输出：
      IMAGE: 生成的图像
      STRING: 状态/日志
    """

    CATEGORY = "Google AI"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "status")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
            },
            "optional": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "file_path": ("STRING", {"multiline": False, "default": ""}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
            },
        }

    def _call_google_ai(
        self,
        api_key: str,
        pil_refs: List[Image.Image],
        prompt_text: str,
    ) -> Tuple[List[Image.Image], str]:
        if not HAS_GOOGLE_AI:
            return [], "未安装 google-genai 库，请先安装：pip install google-genai"
        if not api_key:
            return [], "错误：请输入 Google AI API Key。"

        try:
            # 设置 API Key
            client = genai.Client(api_key=api_key)
            
            # 构建内容
            contents = [prompt_text]
            for pil_ref in pil_refs:
                contents.append(pil_ref)

            # 调用 API
            response = client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE']
                )
            )
            
            # 解析响应
            out_pils = []
            for part in response.parts:
                if part.inline_data is not None:
                    image = part.as_image()
                    if image:
                        out_pils.append(image)
            
            if not out_pils:
                return [], "未从模型收到图片数据。"
            return out_pils, ""
        except Exception as e:
            return [], f"生成图片时出错: {traceback.format_exc()}"

    def generate(
        self,
        api_key: str,
        prompt: str = "",
        file_path: str = "",
        num_images: int = 1,
        image1=None,
        image2=None,
        image3=None,
        image4=None,
        image5=None,
        image6=None,
        image7=None,
        image8=None,
    ):
        all_input_pils: List[Image.Image] = []
        try:
            for img_tensor in [image1, image2, image3, image4, image5, image6, image7, image8]:
                if img_tensor is not None:
                    all_input_pils.extend(_tensor_to_pils(img_tensor))
        except Exception as e:
            return (_pils_to_tensor([]), f"输入图像解析失败：{e}")

        if not all_input_pils:
            return (_pils_to_tensor([]), "错误：请输入至少一张参考图像。")

        # 判定模式
        if not prompt and not file_path:
            return (_pils_to_tensor(all_input_pils), "错误：请输入提示词或提供 CSV/Excel 文件路径。")

        all_out_pils: List[Image.Image] = []
        status_msgs: List[str] = []

        # 单条 prompt
        if prompt:
            total_generated = 0
            for i in range(num_images):
                out_pils, err = self._call_google_ai(api_key, all_input_pils, prompt)
                if err:
                    if i == 0:
                        return (_pils_to_tensor(all_input_pils), err)
                    status_msgs.append(f"第 {i+1} 张图片生成失败：{err}")
                else:
                    all_out_pils.extend(out_pils)
                    total_generated += len(out_pils)
            status_msgs.append(f"已生成 {total_generated} 张图片。")

        # 批量文件
        elif file_path:
            clean_path = file_path.strip()
            
            if (clean_path.startswith('"') and clean_path.endswith('"')) or \
               (clean_path.startswith("'") and clean_path.endswith("'")):
                clean_path = clean_path[1:-1]
            
            import re
            clean_path = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', clean_path)
            clean_path = os.path.normpath(clean_path)
            
            if not os.path.exists(clean_path):
                return (_pils_to_tensor(all_input_pils), f"错误：文件路径不存在: {clean_path}")

            if not HAS_PANDAS:
                return (_pils_to_tensor(all_input_pils), "错误：批量模式需要 pandas，请先安装：pip install pandas openpyxl")

            try:
                if clean_path.lower().endswith(".csv"):
                    try:
                        df = pd.read_csv(clean_path, encoding='utf-8')
                    except UnicodeDecodeError:
                        try:
                            df = pd.read_csv(clean_path, encoding='gbk')
                        except UnicodeDecodeError:
                            df = pd.read_csv(clean_path, encoding='latin1')
                else:
                    df = pd.read_excel(clean_path, sheet_name="Sheet1")
            except Exception as e:
                return (_pils_to_tensor(all_input_pils), f"读取文件失败：{e}")

            if "prompt" not in df.columns:
                return (_pils_to_tensor(all_input_pils), "错误：文件中未找到 'prompt' 列。")

            for idx, row in df.iterrows():
                csv_prompt = row.get("prompt")
                if not isinstance(csv_prompt, str) or not csv_prompt.strip():
                    status_msgs.append(f"第 {idx + 1} 行跳过：空提示词")
                    continue
                out_pils, err = self._call_google_ai(api_key, all_input_pils, csv_prompt)
                if err:
                    status_msgs.append(f"图片 {idx + 1} 生成失败：{err}")
                else:
                    all_out_pils.extend(out_pils)
                    status_msgs.append(f"图片 {idx + 1} 生成成功（{len(out_pils)} 张）。")

            if not all_out_pils:
                return (_pils_to_tensor(all_input_pils), "未从文件中生成任何图片。\n" + "\n".join(status_msgs))

        out_tensor = _pils_to_tensor(all_out_pils)
        
        if len(all_out_pils) > 1:
            sizes = [(pil.width, pil.height) for pil in all_out_pils]
            unique_sizes = list(set(sizes))
            if len(unique_sizes) > 1:
                size_info = f"\n注意：生成了 {len(all_out_pils)} 张不同尺寸的图片 {unique_sizes}，ComfyUI只显示第一张。"
                status = ("\n".join(status_msgs) + size_info) if status_msgs else ("完成" + size_info)
            else:
                status = "\n".join(status_msgs) if status_msgs else "完成"
        else:
            status = "\n".join(status_msgs) if status_msgs else "完成"
            
        return (out_tensor, status)


class GoogleNanoProNode:
    """
    Nano Banana Pro 专用节点 - 使用 Gemini 3 Pro Image Preview 模型
    
    新功能：
    - 支持最多 14 张参考图像（6 张物体 + 5 张人物 + 其他）
    - 支持 1K/2K/4K 分辨率输出
    - 支持多种宽高比设置
    - 支持 Google Search Grounding（实时信息）
    - 内置 Thinking 模式（自动优化构图）
    
    输出：
      IMAGE: 生成的图像
      STRING: 状态/日志
    """

    CATEGORY = "Google AI"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "status")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
            },
            "optional": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "file_path": ("STRING", {"multiline": False, "default": ""}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                # 分辨率设置
                "resolution": (["1K", "2K", "4K"], {"default": "1K"}),
                # 宽高比设置
                "aspect_ratio": (["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"], {"default": "1:1"}),
                # Google Search Grounding
                "enable_google_search": ("BOOLEAN", {"default": False}),
                # 14 个图像输入
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
                "image9": ("IMAGE",),
                "image10": ("IMAGE",),
                "image11": ("IMAGE",),
                "image12": ("IMAGE",),
                "image13": ("IMAGE",),
                "image14": ("IMAGE",),
            },
        }

    def _call_google_ai_pro(
        self,
        api_key: str,
        pil_refs: List[Image.Image],
        prompt_text: str,
        resolution: str,
        aspect_ratio: str,
        enable_google_search: bool,
    ) -> Tuple[List[Image.Image], str]:
        if not HAS_GOOGLE_AI:
            return [], "未安装 google-genai 库，请先安装：pip install google-genai"
        if not api_key:
            return [], "错误：请输入 Google AI API Key。"

        try:
            # 设置 API Key
            client = genai.Client(api_key=api_key)
            
            # 构建内容
            contents = [prompt_text]
            for pil_ref in pil_refs:
                contents.append(pil_ref)

            # 构建配置
            config_params = {
                "response_modalities": ['TEXT', 'IMAGE'],
                "image_config": types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                    image_size=resolution,
                )
            }
            
            # 添加 Google Search Grounding
            if enable_google_search:
                config_params["tools"] = [{"google_search": {}}]
            
            config = types.GenerateContentConfig(**config_params)

            # 调用 API
            response = client.models.generate_content(
                model="gemini-3-pro-image-preview",
                contents=contents,
                config=config
            )
            
            # 解析响应
            out_pils = []
            for part in response.parts:
                if part.inline_data is not None:
                    image = part.as_image()
                    if image:
                        out_pils.append(image)
            
            if not out_pils:
                return [], "未从模型收到图片数据。"
            return out_pils, ""
        except Exception as e:
            return [], f"生成图片时出错: {traceback.format_exc()}"

    def generate(
        self,
        api_key: str,
        prompt: str = "",
        file_path: str = "",
        num_images: int = 1,
        resolution: str = "1K",
        aspect_ratio: str = "1:1",
        enable_google_search: bool = False,
        image1=None,
        image2=None,
        image3=None,
        image4=None,
        image5=None,
        image6=None,
        image7=None,
        image8=None,
        image9=None,
        image10=None,
        image11=None,
        image12=None,
        image13=None,
        image14=None,
    ):
        all_input_pils: List[Image.Image] = []
        try:
            for img_tensor in [image1, image2, image3, image4, image5, image6, image7, 
                               image8, image9, image10, image11, image12, image13, image14]:
                if img_tensor is not None:
                    all_input_pils.extend(_tensor_to_pils(img_tensor))
        except Exception as e:
            return (_pils_to_tensor([]), f"输入图像解析失败：{e}")

        if not all_input_pils:
            return (_pils_to_tensor([]), "错误：请输入至少一张参考图像。")
        
        # 检查图片数量限制
        if len(all_input_pils) > 14:
            return (_pils_to_tensor([]), f"错误：Nano Banana Pro 最多支持 14 张参考图像，当前输入 {len(all_input_pils)} 张。")

        # 判定模式
        if not prompt and not file_path:
            return (_pils_to_tensor(all_input_pils), "错误：请输入提示词或提供 CSV/Excel 文件路径。")

        all_out_pils: List[Image.Image] = []
        status_msgs: List[str] = []
        
        # 添加配置信息到状态
        config_info = f"[Nano Banana Pro] 分辨率: {resolution}, 宽高比: {aspect_ratio}"
        if enable_google_search:
            config_info += ", Google Search: 开启"
        status_msgs.append(config_info)

        # 单条 prompt
        if prompt:
            total_generated = 0
            for i in range(num_images):
                out_pils, err = self._call_google_ai_pro(
                    api_key, all_input_pils, prompt,
                    resolution, aspect_ratio, enable_google_search
                )
                if err:
                    if i == 0:
                        return (_pils_to_tensor(all_input_pils), err)
                    status_msgs.append(f"第 {i+1} 张图片生成失败：{err}")
                else:
                    all_out_pils.extend(out_pils)
                    total_generated += len(out_pils)
            status_msgs.append(f"已生成 {total_generated} 张图片。")

        # 批量文件
        elif file_path:
            clean_path = file_path.strip()
            
            if (clean_path.startswith('"') and clean_path.endswith('"')) or \
               (clean_path.startswith("'") and clean_path.endswith("'")):
                clean_path = clean_path[1:-1]
            
            import re
            clean_path = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', clean_path)
            clean_path = os.path.normpath(clean_path)
            
            if not os.path.exists(clean_path):
                return (_pils_to_tensor(all_input_pils), f"错误：文件路径不存在: {clean_path}")

            if not HAS_PANDAS:
                return (_pils_to_tensor(all_input_pils), "错误：批量模式需要 pandas，请先安装：pip install pandas openpyxl")

            try:
                if clean_path.lower().endswith(".csv"):
                    try:
                        df = pd.read_csv(clean_path, encoding='utf-8')
                    except UnicodeDecodeError:
                        try:
                            df = pd.read_csv(clean_path, encoding='gbk')
                        except UnicodeDecodeError:
                            df = pd.read_csv(clean_path, encoding='latin1')
                else:
                    df = pd.read_excel(clean_path, sheet_name="Sheet1")
            except Exception as e:
                return (_pils_to_tensor(all_input_pils), f"读取文件失败：{e}")

            if "prompt" not in df.columns:
                return (_pils_to_tensor(all_input_pils), "错误：文件中未找到 'prompt' 列。")

            for idx, row in df.iterrows():
                csv_prompt = row.get("prompt")
                if not isinstance(csv_prompt, str) or not csv_prompt.strip():
                    status_msgs.append(f"第 {idx + 1} 行跳过：空提示词")
                    continue
                out_pils, err = self._call_google_ai_pro(
                    api_key, all_input_pils, csv_prompt,
                    resolution, aspect_ratio, enable_google_search
                )
                if err:
                    status_msgs.append(f"图片 {idx + 1} 生成失败：{err}")
                else:
                    all_out_pils.extend(out_pils)
                    status_msgs.append(f"图片 {idx + 1} 生成成功（{len(out_pils)} 张）。")

            if not all_out_pils:
                return (_pils_to_tensor(all_input_pils), "未从文件中生成任何图片。\n" + "\n".join(status_msgs))

        out_tensor = _pils_to_tensor(all_out_pils)
        
        if len(all_out_pils) > 1:
            sizes = [(pil.width, pil.height) for pil in all_out_pils]
            unique_sizes = list(set(sizes))
            if len(unique_sizes) > 1:
                size_info = f"\n注意：生成了 {len(all_out_pils)} 张不同尺寸的图片 {unique_sizes}，ComfyUI只显示第一张。"
                status = ("\n".join(status_msgs) + size_info) if status_msgs else ("完成" + size_info)
            else:
                status = "\n".join(status_msgs) if status_msgs else "完成"
        else:
            status = "\n".join(status_msgs) if status_msgs else "完成"
            
        return (out_tensor, status)


# 注册到 ComfyUI
NODE_CLASS_MAPPINGS = {
    "GoogleNanoNode": GoogleNanoNode,
    "GoogleNanoProNode": GoogleNanoProNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GoogleNanoNode": "Google Nano (Flash)",
    "GoogleNanoProNode": "Google Nano Pro",
}
