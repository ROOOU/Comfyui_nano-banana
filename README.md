# Google Nano - AI图像生成节点

基于 **Google AI API** 的 ComfyUI 自定义节点，支持 Gemini 模型根据参考图像和提示词生成新的图像。

## 🌟 主要功能

### Google Nano (Flash) 节点
- **模型**: Gemini 2.5 Flash Image
- **特点**: 快速生成，低延迟
- **参考图像**: 最多 8 张

### 🆕 Google Nano Pro 节点
- **模型**: Gemini 3 Pro Image Preview (Nano Banana Pro)
- **参考图像**: 最多 14 张
- **分辨率**: 1K / 2K / 4K
- **宽高比**: 10 种预设
- **Google Search**: 支持实时网络信息
- **Thinking模式**: 自动优化构图

## 📋 功能对比

| 功能 | Google Nano (Flash) | Google Nano Pro |
|------|---------------------|-----------------|
| 模型 | gemini-2.5-flash-image | gemini-3-pro-image-preview |
| 参考图像 | 8张 | 14张 |
| 分辨率 | 1K | 1K / 2K / 4K |
| 宽高比 | ❌ | ✅ |
| Google Search | ❌ | ✅ |
| 速度 | 快速 | 较慢（高质量） |

## 🛠️ 安装

### 依赖安装
```bash
pip install google-genai pandas Pillow openpyxl
```

### 插件安装
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ROOOU/Comfyui_nano-banana.git
```

## 📖 使用方法

1. **获取 API Key**: 访问 [Google AI Studio](https://aistudio.google.com/apikey) 获取 API Key

2. **添加节点**: 搜索 "Google Nano"

3. **配置参数**:
   - `api_key`: Google AI API Key
   - `prompt`: 生成提示词
   - `resolution`: 分辨率 (Pro 节点)
   - `aspect_ratio`: 宽高比 (Pro 节点)

4. **连接图像**: 将参考图像连接到 image1-8 (Flash) 或 image1-14 (Pro)

## ⚙️ 参数说明

### 通用参数
| 参数 | 类型 | 说明 |
|------|------|------|
| api_key | STRING | Google AI API Key |
| prompt | STRING | 生成提示词 |
| file_path | STRING | 批量文件路径 |
| num_images | INT | 生成数量 (1-4) |

### Pro 专有参数
| 参数 | 类型 | 说明 |
|------|------|------|
| resolution | ENUM | 1K / 2K / 4K |
| aspect_ratio | ENUM | 宽高比 |
| enable_google_search | BOOL | Google Search Grounding |

## 📄 许可证

MIT License