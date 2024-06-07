import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Créer le répertoire de sortie s'il n'existe pas
output_dir = 'output9'
os.makedirs(output_dir, exist_ok=True)

# Charger les images en couleur
img1 = cv2.imread('im1.jpeg')  # Image gauche
img2 = cv2.imread('im2.jpg')   # Image droite

# Convertir les images en niveaux de gris
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Initialiser le détecteur de points clés
sift = cv2.SIFT_create()

# Détecter les points clés et calculer les descripteurs
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# Matcher les descripteurs avec FLANN
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# Filtrer les correspondances en utilisant l'échelle des descripteurs
good_matches = []
scale_threshold = 0.5  # Seuil d'échelle à ajuster

for m, n in matches:
    if m.distance < 0.75 * n.distance:
        # Récupérer les échelles des descripteurs
        scale1 = keypoints1[m.queryIdx].size
        scale2 = keypoints2[m.trainIdx].size
        # Calculer la différence d'échelle
        scale_ratio = scale1 / scale2 if scale2 != 0 else 0  # Éviter une division par zéro
        # Rejeter si la différence d'échelle dépasse le seuil
        if np.abs(1 - scale_ratio) < scale_threshold:
            good_matches.append(m)

# Dessiner les correspondances
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Ajouter un titre à l'image
title = 'Correspondances avec filtrage par échelle'
cv2.putText(img_matches, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

# Enregistrer et afficher l'image avec les correspondances
output_path_matches = os.path.join(output_dir, 'matches_scale_filtered.png')
cv2.imwrite(output_path_matches, img_matches)
print(f"Image avec correspondances enregistrée dans {output_path_matches}")

# Affichage avec le titre approprié
plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
plt.title(title)
plt.axis('off')
plt.show()
