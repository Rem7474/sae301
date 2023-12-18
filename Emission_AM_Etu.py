#====================================================
# Simulation d'un signal audio modulé/démodulé AM
# Fichier donné aux étudiants
#====================================================

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal                 # à utiliser pour visualiser le signal dans le domaine temporel
import sounddevice as sd

# création du signal Audio
signal, fe = sf.read('NR4.wav', always_2d=True)
# divmod effectue la division de 2 nombres et donne la valeur entière en premier argument et le reste en second
duree=divmod((len(signal)/fe), 60)
print(f"- Le signal audio comprend {len(signal)} échantillons\n- la fréquence d'échantillonnage est {fe} Echan/s\n"
      f"- on a donc une durée de {len(signal)/fe:.0f}s soit {duree[0]:.0f} minutes et {duree[1]:.0f}s")
print("valeurs du signal =",signal)
# Attention, la fonction sf.read() renvoie un tableau de 1 colonne et N lignes
# il faut le transformer en un tableau de 1 ligne et N colonnes (avec la méthode np.ravel()
# np.ravel() pour transformer un tableau multi-dimensionnel en un tableau uni-dimensionnel (1 ligne)
signal2=np.ravel(signal)
print(signal2)

# FFT bilatérale du signal en dBm
# On prend une tranche du signal
signal_trunc=signal2[1000000:1100000]
N=len(signal_trunc)
# fftshift() permet de faire une FFT bilatéral, on décale l'origine au centre
S = 1/N*np.fft.fftshift(np.fft.fft(signal_trunc))
S_mag = np.abs(S)
# pour générer l'axe des fréquences avec un pas de 1
f = np.arange(-fe/2, fe/2, fe/N)
# Création d'une figure avec 2 axes au format 2 lignes 1 colonne
fig, ax = plt.subplots(2,1, figsize=(15, 10))
# Affichage du signal et sa FFT
ax[0].plot(signal_trunc)
ax[1].plot(f, S_mag)
# on affichera le spectre de -20kHz à 20kHz (donc en Bilatéral)
ax[1].set_xlim([-20000,20000])
ax[0].grid()
ax[0].set_title('Signal audio', fontsize=14)
ax[1].grid()
ax[1].set_title('FFT bilatérale du Signal audio (échelle des ordonnées linéaire)', fontsize=14)
# pour afficher la figure
plt.show()
# pour entendre le signal dans le haut-parleur
sd.play(signal2,fe)
print("Fin de la visualisation du signal audio et de sa FFT")