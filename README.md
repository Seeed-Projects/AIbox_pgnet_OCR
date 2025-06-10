# PGNet HAILO Inference (Based on PaddleOCR)

This is PGNET's hailo inference implementation of [hailort](https://github.com/hailo-ai/hailort).

## Setup

```bash
git clone https://github.com/mjq2020/pgnet_ocr.git

cd pgnet_ocr
pip install -r requirements.txt
```

## Run

```bash
# hailo inference
python inference_pgnet.py pgnet_640.hef --camera 0
```

## Result

