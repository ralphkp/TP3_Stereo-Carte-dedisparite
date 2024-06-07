import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random

# Créer le répertoire de sortie s'il n'existe pas
output_dir = 'output9'
os.makedirs(output_dir, exist_ok=True)

# Charger les images en couleur
img1 = cv2.imread('im1.jpeg')  # Image gauche
img2 = cv2.imread('im2.jpg')   # Image droite

# Convertir les images en niveaux de gris
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Initialiser le détecteur de points clés SIFT
sift = cv2.SIFT_create()

# Détecter les points clés et calculer les descripteurs
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# Matcher les descripteurs avec FLANN
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # Paramètres de recherche FLANN

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# Filtrer les correspondances en utilisant l'orientation des descripteurs
good_matches = []
orientation_threshold = 20  # Seuil d'orientation

for m, n in matches:
    if m.distance < 0.75 * n.distance:
        # Récupérer les orientations des descripteurs
        angle1 = keypoints1[m.queryIdx].angle
        angle2 = keypoints2[m.trainIdx].angle
        # Calculer la différence d'angle
        angle_diff = np.abs(angle1 - angle2)
        # Rejeter si la différence d'angle dépasse le seuil
        if angle_diff < orientation_threshold:
            good_matches.append(m)

# Afficher toutes les correspondances filtrées par orientation
img_matches_all = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
title_all = 'Correspondances avec filtrage par orientation'
cv2.putText(img_matches_all, title_all, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
output_path_all = os.path.join(output_dir, 'matches_orientation_filtered_all.png')
cv2.imwrite(output_path_all, img_matches_all)
print(f"Image avec toutes les correspondances enregistrée dans {output_path_all}")

# Affichage avec le titre approprié
plt.imshow(cv2.cvtColor(img_matches_all, cv2.COLOR_BGR2RGB))
plt.title(title_all)
plt.axis('off')
plt.show()

# Limiter à N meilleures correspondances
N = 50  # Par exemple, afficher seulement les 50 meilleures correspondances
limited_matches = sorted(good_matches, key=lambda x: x.distance)[:N]

# Dessiner les correspondances limitées
img_matches_limited = cv2.drawMatches(img1, keypoints1, img2, keypoints2, limited_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
title_limited = f'Top {N} correspondances avec filtrage par orientation'
cv2.putText(img_matches_limited, title_limited, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
output_path_limited = os.path.join(output_dir, f'matches_orientation_filtered_top_{N}.png')
cv2.imwrite(output_path_limited, img_matches_limited)
print(f"Image avec les {N} meilleures correspondances enregistrée dans {output_path_limited}")

# Affichage avec le titre approprié
plt.imshow(cv2.cvtColor(img_matches_limited, cv2.COLOR_BGR2RGB))
plt.title(title_limited)
plt.axis('off')
plt.show()

# Ajuster le seuil de distance et d'orientation
good_matches_strict = []
orientation_threshold_strict = 10  # Seuil d'orientation plus strict
distance_threshold = 0.6    # Seuil de distance plus strict

for m, n in matches:
    if m.distance < distance_threshold * n.distance:
        angle1 = keypoints1[m.queryIdx].angle
        angle2 = keypoints2[m.trainIdx].angle
        angle_diff = np.abs(angle1 - angle2)
        if angle_diff < orientation_threshold_strict:
            good_matches_strict.append(m)

# Dessiner les correspondances avec le filtrage strict
img_matches_strict = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches_strict, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
title_strict = 'Correspondances avec filtrage strict par orientation et distance'
cv2.putText(img_matches_strict, title_strict, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
output_path_strict = os.path.join(output_dir, 'matches_orientation_filtered_strict.png')
cv2.imwrite(output_path_strict, img_matches_strict)
print(f"Image avec correspondances strictement filtrées enregistrée dans {output_path_strict}")

# Affichage avec le titre approprié
plt.imshow(cv2.cvtColor(img_matches_strict, cv2.COLOR_BGR2RGB))
plt.title(title_strict)
plt.axis('off')
plt.show()

# Sélectionner aléatoirement un sous-ensemble de correspondances
subset_size = 50  # Par exemple, visualiser seulement 50 correspondances
random_matches = random.sample(good_matches, min(subset_size, len(good_matches)))

# Dessiner les correspondances aléatoires
img_matches_random = cv2.drawMatches(img1, keypoints1, img2, keypoints2, random_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
title_random = f'Visualisation de {subset_size} correspondances aléatoires'
cv2.putText(img_matches_random, title_random, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
output_path_random = os.path.join(output_dir, 'matches_orientation_filtered_random.png')
cv2.imwrite(output_path_random, img_matches_random)
print(f"Image avec correspondances aléatoires enregistrée dans {output_path_random}")

# Affichage avec le titre approprié
plt.imshow(cv2.cvtColor(img_matches_random, cv2.COLOR_BGR2RGB))
plt.title(title_random)
plt.axis('off')
plt.show()
