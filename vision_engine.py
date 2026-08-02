import torch
import torchvision
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
import numpy as np
from PIL import Image
import cv2

class ObjectDetector:
    def __init__(self, confidence_threshold=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        
        # Load pre-trained SSDLite MobileNetV3
        # V3.22: Optimized for Edge devices (compatible with Hailo architecture)
        self.weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=self.weights).to(self.device)
        self.model.eval()
        
        self.classes = self.weights.meta["categories"]
        self.preprocess = self.weights.transforms()

    def detect(self, image_array):
        """
        Detect objects in an RGB image array.
        Returns a list of detections: {'box': [x1, y1, x2, y2], 'label': 'person', 'score': 0.95}
        """
        if image_array is None:
            return []

        # Convert to PIL then tensor
        img = Image.fromarray(image_array)
        img_t = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.model(img_t)[0]

        detections = []
        for i in range(len(prediction['boxes'])):
            score = prediction['scores'][i].item()
            if score > self.confidence_threshold:
                box = prediction['boxes'][i].cpu().numpy().astype(int).tolist()
                label_id = prediction['labels'][i].item()
                label = self.classes[label_id]
                
                detections.append({
                    'box': box,
                    'label': label,
                    'score': score
                })
        
        return detections

class LineDetector:
    def __init__(self):
        # V3.23: Standard HSV ranges for Yellow and White (tuned for simulators)
        self.yellow_lower = np.array([20, 100, 100])
        self.yellow_upper = np.array([40, 255, 255])
        
        self.white_lower = np.array([0, 0, 180])
        self.white_upper = np.array([180, 50, 255])

    def detect_lines(self, image_array):
        """
        Detect yellow and white lines using traditional CV.
        Returns: {'yellow_lines': [[x1,y1,x2,y2], ...], 'white_lines': [...]}
        """
        if image_array is None:
            return {'yellow': [], 'white': []}

        h, w = image_array.shape[:2]
        # ROI: Focus on bottom 60% of the image
        roi_start = int(h * 0.4)
        roi = image_array[roi_start:, :]
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        
        # 1. Yellow Mask
        y_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        # 2. White Mask
        w_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        
        results = {'yellow': [], 'white': []}
        
        for color, mask in [('yellow', y_mask), ('white', w_mask)]:
            edges = cv2.Canny(mask, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=10)
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # Offset Y coordinate by ROI start
                    results[color].append([int(x1), int(y1 + roi_start), int(x2), int(y2 + roi_start)])
                    
        return results

class ConeDetector:
    def __init__(self):
        # V3.26: HSV Range for Racing Orange
        self.orange_lower = np.array([5, 150, 150])
        self.orange_upper = np.array([15, 255, 255])

    def detect_cones(self, image_array):
        """Detect orange cones using color blobs."""
        if image_array is None: return []
        
        hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.orange_lower, self.orange_upper)
        
        # Clean up mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cones = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100: # Minimum size for a cone
                x, y, w, h = cv2.boundingRect(cnt)
                cones.append({
                    'box': [x, y, x+w, y+h],
                    'label': 'cone',
                    'score': min(1.0, area / 2000.0) # Pseudo-score based on size
                })
        return cones

if __name__ == "__main__":
    # Quick test on a blank image
    detector = ObjectDetector()
    dummy = np.zeros((240, 320, 3), dtype=np.uint8)
    results = detector.detect(dummy)
    print(f"Detection test (blank image): {results}")
