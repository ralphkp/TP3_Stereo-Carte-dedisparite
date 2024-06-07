import cv2
import numpy as np
import os

# Créer le répertoire de sortie s'il n'existe pas
output_dir = 'output4'
os.makedirs(output_dir, exist_ok=True)

# Charger les images
img1 = cv2.imread('im1.jpeg', cv2.IMREAD_GRAYSCALE)  # Image gauche
img2 = cv2.imread('im2.jpg', cv2.IMREAD_GRAYSCALE)   # Image droite

# Initialiser SIFT
sift = cv2.SIFT_create()

# Détecter les points d'intérêt et calculer les descripteurs
keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

# Utiliser FLANN pour matcher les descripteurs
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

# Filtrer les correspondances en utilisant des heuristiques
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# Sélectionner les coordonnées des points correspondants
src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Trouver la matrice de transformation (homographie) entre les images
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Appliquer la transformation perspective à l'image de gauche
img1_warped = cv2.warpPerspective(img1, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))

# Afficher l'image de droite à côté de l'image de gauche transformée
img1_warped[:, 0:img2.shape[1]] = img2

# Enregistrer l'image combinée avec un titre
output_path_combined_warped = os.path.join(output_dir, 'images_warped_combined.png')
cv2.imwrite(output_path_combined_warped, img1_warped)
print(f"Images combinées avec transformation enregistrées dans {output_path_combined_warped}")
