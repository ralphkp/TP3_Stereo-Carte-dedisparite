import cv2
import matplotlib.pyplot as plt
import numpy as np

# Charger les images
img1 = cv2.imread('im1.jpeg')  # Adaptez le chemin
img2 = cv2.imread('im2.jpg')  # Adaptez le chemin

# Vérifier si les images ont été chargées correctement
if img1 is None or img2 is None:
    print("Erreur : vérifiez les chemins des images.")
    exit()

# Convertir en gris
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Initialiser SIFT
sift = cv2.SIFT_create()

# Détecter les points d'intérêt et calculer les descripteurs
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# Création de l'objet BFMatcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Trouver les correspondances
matches = bf.match(descriptors1, descriptors2)

# Trier les correspondances dans l'ordre du meilleur au moins bon
matches = sorted(matches, key=lambda x: x.distance)

# Création de la carte de disparité
disparity_map = np.zeros(gray1.shape, dtype=np.float32)

# Calcul de la disparité pour chaque bonne correspondance
for match in matches[:100]:  # Utiliser les 100 meilleures correspondances
    # Point dans l'image gauche
    x1, y1 = keypoints1[match.queryIdx].pt
    # Point correspondant dans l'image droite
    x2, y2 = keypoints2[match.trainIdx].pt

    # Calcul de la disparité (différence horizontale)
    disparity = x2 - x1

    # Stocker la disparité dans la carte
    disparity_map[int(y1), int(x1)] = disparity

# Normaliser la carte de disparité pour la visualisation
disparity_map_visual = cv2.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Afficher la carte de disparité
plt.figure(figsize=(10, 5))
plt.imshow(disparity_map_visual, cmap='jet')
plt.colorbar()
plt.title('Carte de Disparité')
plt.show()
