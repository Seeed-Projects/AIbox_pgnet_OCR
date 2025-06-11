# PGNet HAILO Inference (Based on PaddleOCR)

This project deploys Baidu's [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) model on Seeed Studio's AI computing box to enable real-time OCR functionality. The project is based on and references [pgnet_ocr](https://github.com/mjq2020/pgnet_ocr) and [hailo_rt](https://github.com/hailo-ai/hailort).

## Hardware prepare

|                                               Raspberry Pi AI box                                              |                                               reComputer R1100                                               |
| :----------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------: |
| ![Raspberry Pi AI Kit](https://media-cdn.seeedstudio.com/media/catalog/product/cache/bb49d3ec4ee05b6f018e93f896b8a25d/i/m/image114993560.jpeg) | ![reComputer R1100](https://media-cdn.seeedstudio.com/media/catalog/product/cache/bb49d3ec4ee05b6f018e93f896b8a25d/2/-/2-114993595-recomputer-ai-industrial-r2135-12.jpg) |
| [**Purchase Now**](https://www.seeedstudio.com/reComputer-AI-R2130-12-p-6368.html?utm_source=PiAICourse&utm_medium=github&utm_campaign=Course) | [**Purchase Now**](https://www.seeedstudio.com/reComputer-AI-Industrial-R2135-12-p-6432.html?utm_source=PiAICourse&utm_medium=github&utm_campaign=Course) |

## Setup

```bash
git clone https://github.com/Seeed-Projects/AIbox_pgnet_OCR.git
cd AIbox_pgnet_OCR
python -m venv .env --system-site-packages
pip install -r requirements.txt
```

## Run

```bash
# hailo inference
python inference_pgnet.py pgnet_640.hef --camera 0
```

## Result

![result][./resource/result.png]
