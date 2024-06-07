import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# Créer le répertoire de sortie s'il n'existe pas
output_dir = 'output2'
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

# Coller les images côte à côte avec des titres
combined_img = np.hstack((img1_sift, img2_sift))

# Ajouter des titres aux images avec les points SIFT détectés
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img1_sift, cmap='gray')
plt.title('Image Gauche avec Points SIFT')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img2_sift, cmap='gray')
plt.title('Image Droite avec Points SIFT')
plt.axis('off')

# Enregistrer les images avec les points SIFT détectés
output_path_img1_sift = os.path.join(output_dir, 'image_gauche_sift_points.png')
plt.savefig(output_path_img1_sift)
print(f"Image gauche avec points SIFT enregistrée dans {output_path_img1_sift}")

output_path_img2_sift = os.path.join(output_dir, 'image_droite_sift_points.png')
plt.savefig(output_path_img2_sift)
print(f"Image droite avec points SIFT enregistrée dans {output_path_img2_sift}")

# Utiliser FLANN pour matcher les descripteurs
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

# Appliquer le ratio test de Lowe pour conserver les bons matches
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# Dessiner les correspondances
img_matches = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Ajouter un titre et enregistrer l'image avec les correspondances
plt.imshow(img_matches)
plt.title('Correspondances SIFT entre les deux images')
plt.axis('off')
output_path_matches = os.path.join(output_dir, 'sift_matches.png')
plt.savefig(output_path_matches)
print(f"Image avec correspondances enregistrée dans {output_path_matches}")
