import numpy as np
import scipy as sc
import cv2
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage import color, exposure, transform
from skimage.exposure import equalize_hist


class ImageCamera:

    def __init__(self, fichier_avec_path=None, image_deja_fft=None):
        self.path_fichier = fichier_avec_path
        self.path_fftdeja = image_deja_fft

        if image_deja_fft is not None:
            self.image_fft = rgb2gray(cv2.imread(f"{self.path_fftdeja}")[:, :, :3])
        else:
            self.image_fft = None

        self.image_ifft = None
        self.image_originale = cv2.imread(f"{self.path_fichier}")  # Image originale
        # self.image_originale = rgb2gray(cv2.imread(f"{self.path_fichier}")[:, :, :3])  # Image originale


    def calcul_fft(self) -> np.ndarray:
        # Calcul de la transformé de fourier
        self.image_fft = np.fft.fftshift(np.fft.fft2(self.image_originale, axes=(0, 1)))
        return self.image_fft


    def calcul_ifft(self) -> np.ndarray:
        # Calcul de la transformé inverse
        # Si l'instance de classe n'a pas d'image fft
        if len(self.image_fft) <= 1 and len(self.image_ifft) <= 1:

            raise ValueError("Il n'y a aucune fft associée à l'objet")

        elif len(self.image_fft) > 1:
            self.image_ifft = np.fft.ifft2(np.fft.ifftshift(self.image_fft))
            return self.image_ifft

        elif len(self.path_fftdeja) > 1:
            self.image_ifft = np.fft.ifft2(np.fft.ifftshift(self.image_deja_fft))
            return self.image_deja_if


    def resize_image(self, taille_voulue) -> None:
        # Fonction qui gère la compression d'image
        pass


    def afficher_fft(self, num=None):
        # Affiche la transformée de fourier de l'image objet
        cv2.imshow("Titir", np.log(abs(self.image_fft)))


    def afficher_ifft(self, num=None) -> None:
        # Affiche la transformé de fourier inverse du array fft
        plt.figure(num=None, figsize=(8, 6), dpi=80)
        plt.imshow((abs(self.image_ifft)), cmap='gray')
        # plt.imshow(np.log(abs(self.image_ifft)), cmap='gray')


    def afficher_image_originale(self) -> None:
        # Affiche l'image originale sans aucun traitement
        plt.figure(num=None, figsize=(8, 6), dpi=80)
        plt.imshow(self.image_originale, cmap='gray')


    def filtre_od(self, force_du_od) -> None:
        # Ici l'idée est de coder un filtre qui permet de damp certaines amplitudes.
        # Il y aurait donc un certain threshold pour lequel on veut baisser l'intensité
        # Mettons de 2. On sait que la source laser est forte au centre.
        # Il suffira donc de voir où est la fréquence la plus présente et appliquer un
        # masque circulaire sur cette zone.
        pass


    def filtre_passe_haut(self, rayon_du_filtre) -> None:

        # Ressemble beaucoup au filtre passe-bas
        # generate spectrum from magnitude image (for viewing only)
        mag = np.abs(self.image_fft)
        spec = np.log(mag) / 20

        radius = rayon_du_filtre
        mask = np.zeros_like(self.image_originale)
        cy = mask.shape[0] // 2
        cx = mask.shape[1] // 2
        cv2.circle(mask, (cx, cy), radius, (255, 255, 255), -1)[0]
        mask = 255 - mask

        mask2 = cv2.GaussianBlur(mask, (19, 19), 0)

        dft_shift_masked = np.multiply(self.image_fft, mask) / 255
        dft_shift_masked2 = np.multiply(self.image_fft, mask2) / 255

        back_ishift = np.fft.ifftshift(self.image_fft)
        back_ishift_masked = np.fft.ifftshift(dft_shift_masked)
        back_ishift_masked2 = np.fft.ifftshift(dft_shift_masked2)

        img_back = np.fft.ifft2(back_ishift, axes=(0, 1))
        img_filtered = np.fft.ifft2(back_ishift_masked, axes=(0, 1))
        img_filtered2 = np.fft.ifft2(back_ishift_masked2, axes=(0, 1))

        # La multiplication par 3 permet d'augmenter la luminosité
        img_back = np.abs(img_back).clip(0, 255).astype(np.uint8)
        img_filtered = np.abs(3 * img_filtered).clip(0, 255).astype(np.uint8)
        img_filtered2 = np.abs(3 * img_filtered2).clip(0, 255).astype(np.uint8)

        cv2.imshow("ORIGINAL", self.image_originale)
        cv2.imshow("SPECTRUM", spec)
        cv2.imshow("MASK", mask)
        cv2.imshow("MASK2", mask2)
        cv2.imshow("ORIGINAL DFT/IFT ROUND TRIP", img_back)
        cv2.imshow("FILTERED DFT/IFT ROUND TRIP", img_filtered)
        cv2.imshow("FILTERED2 DFT/IFT ROUND TRIP", img_filtered2)
        cv2.waitKey(0)


    def filtre_passe_bas(self, rayon_du_filtre) -> None:
        if len(self.image_fft) <= 1:
            raise ValueError("Il faut d'abord faire la fft de l'image avec la fonction calcul_fft")
        else:
            mags = np.abs(self.image_fft)  # Valeur absolue sur les éléments de la fft = amplitudes
            spec = np.log(mags) / 20  # Met les données en échelle logarithmique.

            # On crée maintenant un masque circulaire. Low pass = Masque circulaire blanc au centre
            radius = rayon_du_filtre  # Plus le radius est gros, plus il va overlap sur les hautes fréquences
            mask = np.zeros_like(self.image_originale)
            cy = mask.shape[0] // 2
            cx = mask.shape[1] // 2
            cv2.circle(mask, (cx, cy), radius, (255, 255, 255), -1)[0]

            # On peut ensuite flouter le masque, pour que ces effets soient moins drastiques
            mask2 = cv2.GaussianBlur(mask, (19, 19), 0)

            # Si je comprends bien, on fait une convolution entre la fft et le masque. Selon le type
            # de masque, ça a pour effet de diminuer / enlever certaines fréquences. On a donc fait le
            # masque et, ensuite, on veut le convoluer sur l'image.

            # Application du masque sur l'image
            fft_mask_multiply = np.multiply(self.image_fft, mask) / 255  # Masque brut
            fft_mask2_multiply = np.multiply(self.image_fft, mask2) / 255  # Masque flouté

            # Il faut maintenant revenir à l'image initiale avec ifft
            back_ishift = np.fft.ifftshift(self.image_fft)
            ifft_mask = np.fft.ifftshift(fft_mask_multiply)  # Pas encore sûr de comprendre ifftshift
            ifft_mask2 = np.fft.ifftshift(fft_mask2_multiply)

            # Application du masque
            img_back = np.fft.ifft2(back_ishift, axes=(0, 1))
            img_filtered = np.fft.ifft2(ifft_mask, axes=(0, 1))
            img_filtered2 = np.fft.ifft2(ifft_mask2, axes=(0, 1))

            img_filtered = np.abs(img_filtered).clip(0, 255).astype(np.uint8)
            img_filtered2 = np.abs(img_filtered2).clip(0, 255).astype(np.uint8)

            cv2.imshow("ORIGINAL", self.image_originale)
            cv2.imshow("SPECTRUM", spec)
            cv2.imshow("MASK", mask)
            cv2.imshow("MASK2", mask2)
            cv2.imshow("FILTERED DFT/IFT ROUND TRIP", img_filtered)
            cv2.imshow("FILTERED2 DFT/IFT ROUND TRIP", img_filtered2)
            cv2.waitKey(0)


    def masque_damp(self, threshold: float) -> None:
        pass


    def revert_changes(self):
        pass
