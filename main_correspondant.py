import cv2
import matplotlib.pyplot as plt

# Charger les images
img1 = cv2.imread('im1.jpeg')  # Adaptez le chemin
img2 = cv2.imread('im2.jpg')   # Adaptez le chemin

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
matches = sorted(matches, key = lambda x:x.distance)

# Dessiner les premières 50 correspondances
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Afficher les images avec les points d'intérêt
plt.figure(figsize=(18, 9))
plt.imshow(img_matches)
plt.title('100 meilleures correspondances')
plt.show()
