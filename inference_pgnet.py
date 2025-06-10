import argparse
import os
import time

import cv2
import numpy as np
import onnxruntime
from utils.utils import E2EResizeForTest, KeepKeys, NormalizeImage, ToCHWImage
from e2e_utils.pg_postprocess import PGPostProcess
from hailo.inference_hailo import HailoRTInference
from utils.download import download_model

class PGNetPredictor:
    def __init__(self, model_path, cpu=False):
        self.dict_path = "utils/ic15_dict.txt"
        if not os.path.exists(self.dict_path):
            with open(self.dict_path, "w") as f:
                f.writelines(chr_dct_list)
        if not cpu:
            providers = ["CUDAExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        model_path = download_model(model_path)

        if model_path.endswith(".onnx"):
            self.sess = onnxruntime.InferenceSession(model_path, providers=providers)
        else:
            self.sess = HailoRTInference(model_path)
        self.infer_type = "onnx" if model_path.endswith(".onnx") else "hailo"

    def preprocess(self, img):
        self.ori_im = img.copy()
        # Resize image to match model's expected input size (640x640)
        img = cv2.resize(img, (640, 640))
        data = {
            "image": img,
            "shape": np.array([img.shape[0], img.shape[1], 1.0, 1.0])
        }
        transforms = [
            NormalizeImage(
                scale=1.0 / 255.0,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                order="hwc",
            ),
            ToCHWImage(),
            KeepKeys(keep_keys=["image", "shape"]),
        ]
        for transform in transforms:
            if (
                self.infer_type == "hailo"
                and transform.__class__.__name__ == "NormalizeImage"
            ):
                continue
            data = transform(data)
        
        # Handle both dictionary and list return types from transforms
        if isinstance(data, dict):
            img, shape_list = data["image"], data["shape"]
        else:
            img, shape_list = data[0], data[1]
            
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        return img, shape_list

    def predict(self, img):
        if self.infer_type == "hailo":
            # Ensure input is in the correct format for Hailo
            ort_inputs = {self.sess.get_inputs()[0].name: img.transpose(0, 2, 3, 1).astype(np.uint8)}
        else:
            ort_inputs = {self.sess.get_inputs()[0].name: img}
        outputs = self.sess.run(None, ort_inputs)
        preds = {}
        if isinstance(self.sess, onnxruntime.InferenceSession):
            preds["f_border"] = outputs[0]
            preds["f_char"] = outputs[1]
            preds["f_direction"] = outputs[2]
            preds["f_score"] = outputs[3]
        else:
            for key, output in outputs.items():
                if output.shape[-1] == 4:
                    preds["f_border"] = output.transpose(0, 3, 1, 2).astype(np.float32)
                    preds["f_border"][:, :2, ...] = preds["f_border"][:, :2, ...] / 640
                    preds["f_border"][:, 2:, ...] = preds["f_border"][:, 2:, ...] / 100
                elif output.shape[-1] == 1:
                    preds["f_score"] = output.transpose(0, 3, 1, 2).astype(np.float32)
                elif output.shape[-1] == 2:
                    preds["f_direction"] = output.transpose(0, 3, 1, 2).astype(np.float32)
                elif output.shape[-1] == 37:
                    preds["f_char"] = output.transpose(0, 3, 1, 2).astype(np.float32)
                else:
                    raise ValueError(f"output shape {output.shape} is not supported")
        return preds

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def postprocess(self, preds, shape_list):
        pgpostprocess = PGPostProcess(
            character_dict_path=self.dict_path,
            valid_set="totaltext",
            score_thresh=0.6,
            mode="fast",
        )
        post_result = pgpostprocess(preds, shape_list)
        points, strs = post_result["points"], post_result["texts"]
        dt_boxes = self.filter_tag_det_res_only_clip(points, self.ori_im.shape)
        return dt_boxes, strs

    def process_frame(self, frame):
        img, shape_list = self.preprocess(frame)
        preds = self.predict(img)
        dt_boxes, strs = self.postprocess(preds, shape_list)
        
        # Map boxes back to original image size
        if len(dt_boxes) > 0:  # Only process if boxes were detected
            h_scale = frame.shape[0] / 640.0
            w_scale = frame.shape[1] / 640.0
            dt_boxes = dt_boxes * np.array([w_scale, h_scale])
        
        return dt_boxes, strs

    def draw(self, dt_boxes, strs, frame):
        for box, str in zip(dt_boxes, strs):
            box = box.astype(np.int32).reshape((-1, 1, 2))
            # Draw box in yellow
            cv2.polylines(frame, [box], True, color=(255, 255, 0), thickness=2)
            
            # Calculate font scale based on image width
            font_scale = 1.0  # Increased base font scale
            thickness = 2     # Increased thickness
            
            # Get text size for better positioning
            (text_width, text_height), _ = cv2.getTextSize(
                str, cv2.FONT_HERSHEY_COMPLEX, font_scale, thickness
            )
            
            # Calculate text position to ensure it's visible
            text_x = int(box[0, 0, 0])
            text_y = int(box[0, 0, 1]) - 10  # Move text up a bit from the box
            
            # Draw text with red color
            cv2.putText(
                frame,
                str,
                org=(text_x, text_y),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=font_scale,
                color=(0, 0, 255),  # Red color in BGR
                thickness=thickness
            )
        return frame

def main():
    parser = argparse.ArgumentParser(description="PGPNET inference with USB camera")
    parser.add_argument("model_path", type=str, help="onnxmodel path")
    parser.add_argument("--cpu", action="store_true", help="cpu inference, default device is gpu")
    parser.add_argument("--camera", type=int, default=0, help="camera device index")
    args = parser.parse_args()

    # Initialize camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Initialize predictor
    predictor = PGNetPredictor(args.model_path, args.cpu)

    print("Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Process frame
        dt_boxes, strs = predictor.process_frame(frame)
        
        # Draw results
        frame = predictor.draw(dt_boxes, strs, frame)
        
        # Display FPS
        cv2.putText(frame, f"Text detected: {len(strs)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show frame
        cv2.imshow('PGNet OCR', frame)

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
