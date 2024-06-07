import cv2
import numpy as np
import os

# Charger les images en niveau de gris
img1 = cv2.imread('im1.jpeg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('im2.jpg', cv2.IMREAD_GRAYSCALE)

# Initialiser le détecteur de points clés (SIFT)
sift = cv2.SIFT_create()

# Détecter les points clés et calculer les descripteurs
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# Matcher les descripteurs avec FLANN
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# Filtrer les bonnes correspondances en utilisant l'analyse des directions
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        # Obtenir les coordonnées des points d'intérêt correspondants
        pt1 = keypoints1[m.queryIdx].pt
        pt2 = keypoints2[m.trainIdx].pt
        # Calculer le vecteur entre les points d'intérêt
        vector = np.array(pt2) - np.array(pt1)
        # Calculer l'angle (direction) du vecteur
        angle = np.arctan2(vector[1], vector[0]) * 180 / np.pi
        # Filtrer les correspondances avec un écart angulaire supérieur à une certaine valeur
        if abs(angle) < 300:  # Vous pouvez ajuster cette valeur en fonction de votre cas d'utilisation
            good_matches.append(m)

# Dessiner les correspondances filtrées
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Enregistrer l'image avec les correspondances filtrées
output_dir = 'output_filtered_matches'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'filtered_matches.png')
cv2.imwrite(output_path, img_matches)
print(f"Image avec correspondances filtrées enregistrée dans {output_path}")
