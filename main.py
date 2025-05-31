import cv2
from fer import FER
import pygame
import numpy as np
import os

detector = FER(mtcnn=True)


pygame.mixer.init()


cam = cv2.VideoCapture(0)
print("📸 Camera is open — press SPACE to capture image")
  

while True:
    ret, frame = cam.read()
    cv2.imshow('press the space to capture', frame)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        cv2.imwrite("captured.jpg", frame)
        print("✅ Image saved as captured.jpg")
        break

cam.release()
cv2.destroyAllWindows()


img = cv2.imread("captured.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_float = np.asarray(img_rgb).astype(np.float32)


result = detector.top_emotion(img_float)
emotion = result[0] if result else "neutral"
print(f"😊 Detected Emotion: {emotion}")


if emotion == "happy":
    song_path = "songs/happy.mp3"
elif emotion == "sad":
    song_path = "songs/sad.mp3"
else:
    song_path = "songs/neutral.mp3"

song_name = os.path.basename(song_path)


display_frame = cv2.imread("captured.jpg")
cv2.putText(display_frame, f"Emotion: {emotion}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.putText(display_frame, f"Playing: {song_name}", (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
cv2.imshow("🎵 Mood-Based Music Player", display_frame)
cv2.waitKey(3000)
cv2.destroyAllWindows()


print(f"🎶 Playing song for: {emotion} - Now playing: {song_name}")
pygame.mixer.music.load(song_path)
pygame.mixer.music.play()

print("🎧 Press 's' to stop the music.")


while pygame.mixer.music.get_busy():
    if cv2.waitKey(1) & 0xFF == ord('s'):  
        pygame.mixer.music.stop() 
        print("⏹ Music stopped by user.")
        break

pygame.quit()  