import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Créer le répertoire de sortie s'il n'existe pas
output_dir = 'output_inverted_images'
os.makedirs(output_dir, exist_ok=True)

# Charger les images en couleur (inversées)
img1 = cv2.imread('im2.jpg')  # Image gauche
img2 = cv2.imread('im1.jpeg')  # Image droite

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

# Filtrer les correspondances en utilisant la géométrie épipolaire pour traiter les changements de perspective
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        pt1 = keypoints1[m.queryIdx].pt
        pt2 = keypoints2[m.trainIdx].pt
        if np.abs(pt1[1] - pt2[1]) < 10:  # Filtrer les correspondances sur la même ligne verticale
            good_matches.append(m)

# Dessiner les correspondances
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Ajouter un titre à l'image
title = 'Correspondances avec images inversées'
cv2.putText(img_matches, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

# Enregistrer et afficher l'image avec les correspondances
output_path_matches = os.path.join(output_dir, 'matches_inverted_images.png')
cv2.imwrite(output_path_matches, img_matches)
print(f"Image avec correspondances enregistrée dans {output_path_matches}")

# Affichage avec le titre approprié
plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
plt.title(title)
plt.axis('off')
plt.show()

# Calculer les informations 3D
# Note: Cette partie nécessite une triangulation et une reconstruction 3D à partir des correspondances,
# ce qui n'est pas implémenté dans ce code d'exemple.
