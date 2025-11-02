#!/usr/bin/env python3

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time


class SignLanguageCNN(nn.Module):
    def __init__(self, num_classes=24):
        super(SignLanguageCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.25)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.25)

        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dropout4 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, 128)
        self.bn8 = nn.BatchNorm1d(128)
        self.dropout5 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.bn7(self.fc1(x)))
        x = self.dropout4(x)

        x = F.relu(self.bn8(self.fc2(x)))
        x = self.dropout5(x)

        x = self.fc3(x)

        return x


class NewSignLanguagePredictor:
    def __init__(self, model_path='model.pth'):
        self.label_map = {}

        for i in range(9):
            self.label_map[i] = chr(65 + i)

        for i in range(10, 25):
            self.label_map[i] = chr(65 + i)

        self.device = self._setup_device()

        self.model = self._load_model(model_path)

        print(f"🆕 NEW Sign Language Predictor initialized")
        print(f"   Device: {self.device}")
        print(f"   Model: {model_path}")
        print(f"   Classes: {len(self.label_map)} (NEW clean architecture)")
        print(f"   Available letters: {sorted(self.label_map.values())}")

        self._verify_mapping()

    def _setup_device(self):
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            device = torch.device('xpu:0')
            print(f"✅ Using Intel Arc GPU: {torch.xpu.get_device_name(0)}")
        elif torch.cuda.is_available():
            device = torch.device('cuda:0')
            print(f"✅ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print("⚠️  Using CPU")
        return device

    def _load_model(self, model_path):
        model = SignLanguageCNN(num_classes=24).to(self.device)

        try:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            print(f"✅ NEW Model loaded successfully from {model_path}")

            dummy_input = torch.randn(1, 1, 28, 28).to(self.device)
            with torch.no_grad():
                output = model(dummy_input)
                expected_shape = torch.Size([1, 24])
                if output.shape == expected_shape:
                    print(f"✅ NEW Model output shape verified: {output.shape}")
                else:
                    print(f"⚠️  Model output shape mismatch: expected {expected_shape}, got {output.shape}")

        except Exception as e:
            print(f"❌ Error loading NEW model: {e}")
            print(f"   Make sure {model_path} exists and was trained with NEW 24-class architecture")
            raise

        return model

    def _verify_mapping(self):
        issues = []

        if 'J' in self.label_map.values():
            issues.append("J should not be present (requires motion)")
        if 'Z' in self.label_map.values():
            issues.append("Z should not be present (not in dataset)")
        if self.label_map.get(24) != 'Y':
            issues.append(f"Label 24 should be Y, got {self.label_map.get(24)}")
        if len(self.label_map) != 24:
            issues.append(f"Should have 24 classes, got {len(self.label_map)}")

        if issues:
            print(f"   ⚠️  NEW Mapping issues: {issues}")
        else:
            print(f"   ✅ NEW Label mapping verified correct")

    def preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

        normalized = resized.astype(np.float32) / 255.0

        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)

        return tensor.to(self.device)

    def predict(self, frame):
        input_tensor = self.preprocess_frame(frame)

        processed_img = input_tensor.squeeze().cpu().numpy()
        processed_img = (processed_img * 255).astype(np.uint8)

        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

            predicted_class = predicted_class.item()
            confidence = confidence.item()

            if predicted_class in self.label_map:
                predicted_letter = self.label_map[predicted_class]
            else:
                predicted_letter = "?"

        return predicted_letter, confidence, processed_img

    def run_webcam(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return

        print("\n🎥 Starting NEW webcam application...")
        print("📝 NEW Instructions:")
        print("   - Position your hand in the center square")
        print("   - Press 'q' to quit")
        print("   - Press SPACEBAR or 's' to take screenshot")
        print("   🆕 NEW: Available letters: A-I, K-Y (24 letters)")
        print("   🆕 NEW: Missing: J (requires motion), Z (not in dataset)")

        fps_counter = 0
        fps_start_time = time.time()
        fps_display = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error reading frame")
                    break

                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                roi_size = 300
                roi_x = (w - roi_size) // 2
                roi_y = (h - roi_size) // 2

                roi = frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]

                predicted_letter, confidence, processed_img = self.predict(roi)

                cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_size, roi_y+roi_size), (255, 0, 0), 6)

                prediction_text = f"Letter: {predicted_letter}"
                confidence_text = f"Confidence: {confidence*100:.1f}%"

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.2
                thickness = 2

                if confidence > 0.8:
                    color = (0, 255, 0)
                elif confidence > 0.5:
                    color = (0, 255, 255)
                else:
                    color = (0, 0, 255)

                cv2.putText(frame, prediction_text, (50, 50), font, font_scale, color, thickness)
                cv2.putText(frame, confidence_text, (50, 100), font, 0.8, color, 2)

                cv2.putText(frame, "🆕 NEW 24-Class Model", (50, 150), font, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, "Letters: A-I, K-Y (No J, No Z)", (50, 180), font, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, "Clean Architecture", (50, 200), font, 0.4, (255, 255, 255), 1)

                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    fps_display = fps_counter
                    fps_counter = 0
                    fps_start_time = time.time()

                cv2.putText(frame, f"FPS: {fps_display}", (50, h-50), font, 0.7, (255, 255, 255), 2)

                cv2.putText(frame, "Press 'q' to quit, SPACEBAR/'s' to save", (50, h-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                processed_display = cv2.resize(processed_img, (100, 100))
                processed_display_3ch = cv2.cvtColor(processed_display, cv2.COLOR_GRAY2BGR)
                frame[10:110, w-110:w-10] = processed_display_3ch
                cv2.rectangle(frame, (w-110, 10), (w-10, 110), (0, 255, 255), 2)
                cv2.putText(frame, "NEW Processed", (w-108, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                cv2.imshow('🆕 NEW Sign Language Recognition', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') or key == 32:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"sign_prediction_NEW_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 NEW Frame saved as {filename}")

        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("🛑 NEW Webcam stopped")


def main():
    try:
        predictor = NewSignLanguagePredictor()

        predictor.run_webcam()

    except FileNotFoundError:
        print("❌ Error: NEW model file not found")
        print("   Expected: 'best_sign_language_model_pytorch_corrected_24classes.pth'")
        print("   Please train the NEW model using the corrected notebook first")
        print("   Or rename your corrected model file to the expected name")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
