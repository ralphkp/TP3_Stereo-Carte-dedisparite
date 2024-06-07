import cv2
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Spécifiez les chemins complets vers les images
    img1_path = 'im1.jpeg'
    img2_path = 'im2.jpg'

    # Charger les images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Vérifiez si les images ont été chargées correctement
    if img1 is None or img2 is None:
        print("Erreur : vérifiez les chemins des images.")
        exit()

    # Convertir en gris et détecter les points SIFT
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # Calcul et filtrage des correspondances
    matches = filter_matches_by_angle(keypoints1, keypoints2, descriptors1, descriptors2)

    # Visualiser les correspondances filtrées
    visualize_matches(img1, keypoints1, img2, keypoints2, matches)

def filter_matches_by_angle(keypoints1, keypoints2, descriptors1, descriptors2):
    # Création de l'objet BFMatcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Appliquer le test de ratio de Lowe
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Définition de la fonction pour calculer l'angle
    def calculate_angle(pt1, pt2):
        x_diff = pt2[0] - pt1[0]
        y_diff = pt2[1] - pt1[1]
        angle = np.arctan2(y_diff, x_diff) * 180 / np.pi  # Convertir en degrés
        return angle

    # Filtrer les correspondances basées sur l'angle
    filtered_matches = []
    for match in good_matches:
        pt1 = keypoints1[match.queryIdx].pt
        pt2 = keypoints2[match.trainIdx].pt
        angle = calculate_angle(pt1, pt2)
        
        # Accepter la correspondance si l'angle est proche de 0 ou 180 degrés
        if abs(angle) < 10 or abs(angle - 180) < 10:  # Seuil d'angle de 10 degrés
            filtered_matches.append(match)
    
    return filtered_matches

def visualize_matches(img1, keypoints1, img2, keypoints2, matches):
    # Dessiner les correspondances filtrées
    img_filtered_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Affichage
    plt.figure(figsize=(18, 9))
    plt.imshow(img_filtered_matches)
    plt.title('Correspondances Filtrées par Angle')
    plt.show()

if __name__ == "__main__":
    main()
