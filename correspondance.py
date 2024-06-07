import cv2
import numpy as np
import matplotlib.pyplot as plt

# Charger les images
img1 = cv2.imread('im1.jpeg', cv2.IMREAD_GRAYSCALE)  # Adaptez le chemin
img2 = cv2.imread('im2.jpg', cv2.IMREAD_GRAYSCALE)  # Adaptez le chemin

# Initialiser SIFT
sift = cv2.SIFT_create()

# Détecter les points d'intérêt et calculer les descripteurs
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# Configurer FLANN Matcher
index_params = dict(algorithm = 1, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Trouver les correspondances
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# Filtrer les correspondances en utilisant le test de Lowe
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# Préparation des points pour RANSAC
if len(good_matches) > 4:
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Calculer la matrice fondamentale avec RANSAC
    M, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC)

    # Sélectionner seulement les correspondances inliers
    matchesMask = mask.ravel().tolist()
    inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]]

    # Dessiner seulement les inliers
    img_inlier_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, inlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
else:
    print("Pas assez de correspondances - RANSAC a échoué")
    matchesMask = None

# Afficher les résultats RANSAC
if matchesMask is not None:
    plt.figure(figsize=(12, 6))
    plt.imshow(img_inlier_matches)
    plt.title('Correspondances inliers après RANSAC')
    plt.show()
else:
    plt.figure(figsize=(12, 6))
    plt.imshow(img_matches)  # Afficher les correspondances initiales si RANSAC échoue
    plt.title('Correspondances avant RANSAC')
    plt.show()
