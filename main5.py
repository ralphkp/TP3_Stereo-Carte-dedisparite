import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Créer le répertoire de sortie s'il n'existe pas
output_dir = 'output5'
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

# Filtrer les correspondances en utilisant l'heuristique
good_matches = []
for m, n in matches:
    if m.distance < n.distance:
        good_matches.append(m)

# Dessiner les correspondances
img_matches = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Ajouter un titre à l'image
title = 'Correspondances SIFT avec heuristique'
cv2.putText(img_matches, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

# Enregistrer et afficher l'image avec les correspondances
output_path_matches = os.path.join(output_dir, 'sift_matches.png')
cv2.imwrite(output_path_matches, img_matches)
print(f"Image avec correspondances enregistrée dans {output_path_matches}")

# Affichage avec le titre approprié
plt.imshow(img_matches)
plt.title(title)
plt.axis('off')
plt.show()
