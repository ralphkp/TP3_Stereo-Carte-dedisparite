import cv2
import matplotlib.pyplot as plt

# Charger les images
img1 = cv2.imread('im1.jpeg')  # Adaptez le chemin
img2 = cv2.imread('im2.jpg')  # Adaptez le chemin

# Convertir en gris
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Initialiser SIFT
sift = cv2.SIFT_create()

# Détecter les points d'intérêt et calculer les descripteurs
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# Dessiner les points d'intérêt
img1_keypoints = cv2.drawKeypoints(gray1, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_keypoints = cv2.drawKeypoints(gray2, keypoints2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Afficher les images avec les points d'intérêt
plt.figure(figsize=(14, 7))
plt.subplot(121)
plt.imshow(img1_keypoints)
plt.title('Points d\'intérêt Image 1')
plt.subplot(122)
plt.imshow(img2_keypoints)
plt.title('Points d\'intérêt Image 2')
plt.show()
