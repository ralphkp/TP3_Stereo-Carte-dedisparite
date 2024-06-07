import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# Créer le répertoire de sortie s'il n'existe pas
output_dir = 'output1'
os.makedirs(output_dir, exist_ok=True)

# Charger les images
img1 = cv2.imread('im1.jpeg', cv2.IMREAD_GRAYSCALE)  # Image gauche
img2 = cv2.imread('im2.jpg', cv2.IMREAD_GRAYSCALE)   # Image droite

# Initialiser SIFT
sift = cv2.SIFT_create()

# Détecter les points d'intérêt et calculer les descripteurs
keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

# Dessiner les points détectés sur les images
img1_sift = cv2.drawKeypoints(img1, keypoints_1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_sift = cv2.drawKeypoints(img2, keypoints_2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Coller les images côte à côte
img_combined = np.hstack((img1_sift, img2_sift))

# Ajouter des titres et afficher les images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img1_sift, cmap='gray')
plt.title('Image Gauche avec Points SIFT')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img2_sift, cmap='gray')
plt.title('Image Droite avec Points SIFT')
plt.axis('off')

# Enregistrer le résultat
output_path = os.path.join(output_dir, 'sift_points_combined.png')
plt.savefig(output_path)
plt.show()

print(f"Image combinée enregistrée dans {output_path}")
