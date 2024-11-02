import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self, max_num_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.cap = cv2.VideoCapture(0)

    def get_hand_landmarks(self):
        success, frame = self.cap.read()
        if not success:
            print("Não foi possível capturar imagem da câmera.")
            return None

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        hand_landmarks_list = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                hand_landmarks_list.extend(landmarks)

        return hand_landmarks_list


    def run(self):
        while self.cap.isOpened():
            hand_landmarks = self.get_hand_landmarks()
            
            # Sai do loop se a tecla 'q' for pressionada
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Libera a câmera e fecha as janelas
        self.cap.release()
        cv2.destroyAllWindows()

# # Exemplo de uso
# if __name__ == "__main__":
#     hand_tracker = HandTracker()
#     hand_tracker.run()
