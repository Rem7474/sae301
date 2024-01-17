#====================================================
# Simulation d'un signal audio modulé/démodulé AM
# Fichier donné aux étudiants
#====================================================

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal                 # à utiliser pour visualiser le signal dans le domaine temporel
import sounddevice as sd
import math

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
plt.show()
# pour entendre le signal dans le haut-parleur
sd.play(signal2,fe)



# Etape 1 : Modulation du signal audio AM
#====================================================
# conversion en dBm
S_eff=S_mag/np.sqrt(2)
S_dbm=10*np.log10(np.square(S_eff)/50*1000)
# Création d'un signal sinusoïdal de fréquence fp et d'amplitude Ap=1V
fp=10000
Ap=1
# Calcul des échantillons du signal modulé
# On multiplie le signal audio par le signal sinusoïdal
signal_porteuse=Ap*np.sin(2*np.pi*fp*np.arange(len(signal_trunc))/fe)
signal_mod=signal_trunc*signal_porteuse

S = 1/N*np.fft.fftshift(np.fft.fft(signal_mod))
S_mag = np.abs(S)
# conversion en dBm
S_eff=S_mag/np.sqrt(2)
S_mod_dbm=10*np.log10(np.square(S_eff)/50*1000)


# FIGURE 1 : Affichage du signal de la porteuse et du signal modulé AM
# Création d'une figure avec 2 axes au format 2 lignes 1 colonne
fig, ax = plt.subplots(2,1, figsize=(15, 10))
# Affichage du signal de la porteuse et le signal modulé AM
ax[0].plot(signal_porteuse[0:1000])
ax[1].plot(signal_trunc[0:1000])
ax[1].plot(signal_mod[0:1000])
#affichage de la graduation en secondes sur l'axe des abscisses (conversion en secondes)
graduation=np.arange(0,1000,1000/fe/6)
#arrondi à 3 chiffres après la virgule
graduation=np.around(graduation,3)
#affichage de la graduation
#ax[0].set_xticks([])
ax[0].set_xticklabels(graduation)
#ax[1].set_xticks([])
ax[1].set_xticklabels(graduation)
ax[0].grid()
ax[0].set_title('Signal de la porteuse', fontsize=14)
ax[1].grid()
ax[1].set_title('Signal audio et signal modulé AM', fontsize=14)

 
# FIGURE 2 : Affichage de la FFT du signal audio et du signal modulé AM
# Création d'une figure avec 2 axes au format 2 lignes 1 colonne
fig, ax = plt.subplots(2,1, figsize=(15, 10))
# Affichage de la FFT du signal audio et du signal modulé AM
ax[0].plot(f, S_dbm)
ax[1].plot(f, S_mod_dbm)
# on affichera le spectre de -20kHz à 20kHz (donc en Bilatéral)
ax[1].set_xlim([-20000,20000])
ax[0].set_xlim([-20000,20000])
ax[0].grid()
ax[0].set_title('DSP bilatérale du Signal audio', fontsize=14)
ax[1].grid()
ax[1].set_title('DSP bilatérale du Signal audio modulé AM', fontsize=14)


# Etape 2 : Démodulation synchrone du signal audio AM
#====================================================
# On multiplie le signal audio modulé par le signal sinusoïdal
signal_dem1=signal_mod*np.sin(2*np.pi*fp*np.arange(len(signal_trunc))/fe)

#affichage du signal modulé AM et du signal démodulé AM
fig, ax = plt.subplots(2,1, figsize=(15, 10))

# FIGURE 3 : Affichage du signal audio modulé AM et du signal audio multiplié par la porteuse
ax[0].plot(signal_mod)
ax[1].plot(signal_dem1)
ax[0].set_xlim([0,1000])
ax[1].set_xlim([0,1000])
ax[0].grid()
ax[0].set_title('Signal audio modulé AM', fontsize=14)
ax[1].grid()
ax[1].set_title('Signal audio multiplié par la porteuse', fontsize=14)

# calcul de la DSP du signal démodulé
S = 1/N*np.fft.fftshift(np.fft.fft(signal_dem1))
S_dem1_abs = np.abs(S)
# conversion en dBm
S_eff=S_dem1_abs/np.sqrt(2)
S_dem1_dbm=10*np.log10(np.square(S_eff)/50*1000)



# FIGURE 4 : Affichage de la DSP du signal audio et du signal audio démodulé
fig, ax = plt.subplots(2,1, figsize=(15, 10))
ax[0].plot(f, S_dbm)
ax[1].plot(f, S_dem1_dbm)
# on affichera le spectre de -20kHz à 20kHz (donc en Bilatéral)
ax[0].set_xlim([-30000,30000])
ax[1].set_xlim([-30000,30000])
ax[0].grid()
ax[0].set_title('DSP du signal audio', fontsize=14)
ax[1].grid()
ax[1].set_title('DSP du Signal audio démodulé', fontsize=14)

# filtrage du signal démodulé avec un filtre de butterworth
# on utilise la fonction scipy.signal.butter()
ordre=8
fc=10000
b, a = scipy.signal.butter(ordre, fc, 'low', fs=fe, output='ba')
signal_dem2 = scipy.signal.lfilter(b, a, signal_dem1)
# calcul de la DSP du signal démodulé
S = 1/N*np.fft.fftshift(np.fft.fft(signal_dem2))
S_dem2_abs = np.abs(S)
# conversion en dBm
S_eff=S_dem2_abs/np.sqrt(2)
S_dem2_dbm=10*np.log10(np.square(S_eff)/50*1000)


# FIGURE 5 : Affichage de la DSP du signal audio démodulé avec le filtre passe-bas
fig, ax = plt.subplots(1,1, figsize=(15, 10))
#affichage des deux signaux
ax.plot(f,S_dem2_dbm)
ax.set_xlim([-30000,30000])
ax.grid()
ax.set_title('DSP du Signal audio démodulé avec filtre passe-bas', fontsize=14)

# FIGURE 6 : Affichage du signal audio et du signal démodulé
# affichage du signal dem1 et dem2 sur la même figure (1seul figure)
fig, ax = plt.subplots(1,1, figsize=(15, 10))
#affichage des deux signaux
ax.plot(signal_trunc)
ax.plot(signal_dem2)
ax.grid()
ax.set_title('Signal audio démodulé', fontsize=14)

#FIGURE 7 : Effet d'une erreur de synchronisation de 0.1Hz
fp_erreur=fp+0.1
signal_dem_erreur=signal_mod*np.sin(2*np.pi*fp_erreur*np.arange(len(signal_trunc))/fe)
# filtrage du signal démodulé avec un filtre de butterworth
# on utilise la fonction scipy.signal.butter()
ordre=8
fc=10000
b, a = scipy.signal.butter(ordre, fc, 'low', fs=fe, output='ba')
signal_dem_erreur_lowpass = scipy.signal.lfilter(b, a, signal_dem_erreur)

fig, ax = plt.subplots(1,1, figsize=(15, 10))
#affichage des deux signaux
ax.plot(signal_dem2)
ax.plot(signal_dem_erreur_lowpass)
ax.grid()
ax.set_title('Signal audio démodulé avec erreur de synchronisation de 0.1Hz', fontsize=14)

#affichage du graphique
plt.show()

