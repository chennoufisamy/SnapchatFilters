import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import random

# Charger les detecteur de visage, d'yeux et de corps
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye_tree_eyeglasses.xml')
body_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_upperbody.xml')


# Charger les images et changer le canal bleu et rouge en gardant le canal alpha

img_chapeau = cv2.imread('images/chapeau.png', cv2.IMREAD_UNCHANGED)
b, g, r, a = cv2.split(img_chapeau)
img_chapeau = cv2.merge([r, g, b, a])

img_barbe = cv2.imread('images/barbe.png', cv2.IMREAD_UNCHANGED)
b, g, r, a = cv2.split(img_barbe)
img_barbe = cv2.merge([r, g, b, a])

img_coeur = cv2.imread('images/coeur.png', cv2.IMREAD_UNCHANGED)
b, g, r, a = cv2.split(img_coeur)
img_coeur = cv2.merge([r, g, b, a])

# Initialiser les variables pour les activations/desactivations des filtres et initialiser le frame
frame = None
filtreGlassesActiveB = False
filtreSepiaActiveB = False
filtreCoeursActiveB = False


# Fonction pour activer/desactiver les filtres
def filtreGlassesActive():
    global filtreGlassesActiveB
    filtreGlassesActiveB = not filtreGlassesActiveB
    
def filtreSepiaActive():
    global filtreSepiaActiveB
    filtreSepiaActiveB = not filtreSepiaActiveB
    
def filtreCoeursActive():
    global filtreCoeursActiveB
    filtreCoeursActiveB = not filtreCoeursActiveB
    
# Fonction pour appliquer le filtre pere noel
def pereNoel(face_cascade,frame,img_chapeau,img_barbe) :
    
    # Convertir l'image en niveaux de gris pour la détection du corps
    grayMultiple = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detection du visage
    facesMultiple = face_cascade.detectMultiScale(grayMultiple, 1.1, 4)

    if np.size(facesMultiple)>0:
        #Couper la video et recuperer la partie du visage
        for (x,y,w,h) in facesMultiple:
            
            # Coordonnées
            w2=w
            h2=h
            hx1=x 
            hx2=hx1+w2
            hy1=y
            hy2=hy1+h2
            
            # Redimensionner les images        
            img_chapeau_resized = cv2.resize(img_chapeau, (hx2-hx1,hy2-hy1))
            img_barbe_resized = cv2.resize(img_barbe, (hx2-hx1,hy2-hy1))
            
            # Récupérer les dimensions du cadre
            chapeau_height, chapeau_width = img_chapeau_resized.shape[:2]
            barbe_height, barbe_width = img_barbe_resized.shape[:2]
            
            # Adjuster les coordonnées pour le chapeau
            hy1_chapeau = hy1 - chapeau_height + 60
            
            # Adjuster les coordonnées pour la barbe
            hy1_barbe = hy1 + h - barbe_height + 100
            
            if hy1_chapeau >= 0:  # Assurer que le chapeau est dans le cadre
                
                # Recuperer le canal alpha
                alpha_channel = img_chapeau_resized[:, :, 3] / 255.0
                
                # Ajouter une dimension
                alpha_channel = np.expand_dims(alpha_channel, axis=2)
                
                # Calculer le resultat
                result = alpha_channel * img_chapeau_resized[:, :, :3] + (1.0 - alpha_channel) * frame[hy1_chapeau:hy1_chapeau+chapeau_height, hx1:hx1+chapeau_width]
                
                # Mettre à jour le cadre
                frame[hy1_chapeau:hy1_chapeau+chapeau_height, hx1:hx1+chapeau_width] = result.astype(np.uint8)
            
            if hy1_barbe >= 0:  # Assurer que la barbe est dans le cadre
                
                # Recuperer le canal alpha
                alpha_channel = img_barbe_resized[:, :, 3] / 255.0
                
                # Ajouter une dimension
                alpha_channel = np.expand_dims(alpha_channel, axis=2)
                
                # Calculer le resultat
                result = alpha_channel * img_barbe_resized[:, :, :3] + (1.0 - alpha_channel) * frame[hy1_barbe:hy1_barbe + barbe_height, hx1:hx1 + barbe_width]
                
                # Mettre à jour le cadre
                frame[hy1_barbe:hy1_barbe+barbe_height, hx1:hx1+barbe_width] = result.astype(np.uint8)

# Fonction pour appliquer le filtre sepia
def sepia(frame):
    # Conversion en sépia
    sepia_filter = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])

    sepia_frame = cv2.transform(frame[0:frame.shape[0],0:frame.shape[1]], sepia_filter)

    frame[0:frame.shape[0],0:frame.shape[1]] = sepia_frame


#Initialisation image des coeurs et leur position initiale pour les appliquer au fond
coeurs = [(0, i * 60) for i in range(10)]
coeur_y = 0

# Fonction pour appliquer le filtre coeurs
def fondCoeurs(body_cascade,frame,coeur):
    
    grayMultiple = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bodyDetection = body_cascade.detectMultiScale(grayMultiple, 1.1, 4)

    # Récupérer les dimensions du cadre
    height, width, _ = frame.shape
    coeur = cv2.resize(coeur, (90, 103))
    
    # Initialiser l'overlay
    overlay = np.zeros_like(frame)
    
    # Initialiser un masque vide
    bodyMask = np.zeros_like(grayMultiple, dtype=bool)

    for (x, y, w, h) in bodyDetection:
        
        # DEfinir le pourcentage par lequel vous souhaitez réduire la taille du rectangle
        reduction_percentage = 0.2  # 20% reduction

        # Calculer la quantité par laquelle réduire les dimensions du rectangle
        w_reduction = int(w * reduction_percentage)
        h_reduction = int(h * reduction_percentage)

        # Ajuster les dimensions du rectangle
        bodyMask[y+h_reduction:y+h-h_reduction, x+w_reduction:x+w-w_reduction] = True  

    # Superposer les flocons de neige sur l'image du cadre de la webcam
    for i, (coeur_y, snow_x) in enumerate(coeurs):
        coeur_y = (coeur_y + random.randint(1, 5)) % height # Faire bouger les coeurs
        coeurs[i] = (coeur_y, snow_x)
        coeur_rgb = coeur[:, :, :3]

        # Creer un masque où True représente les pixels du coeur qui sont sur le corps
        coeur_mask = bodyMask[coeur_y:coeur_y+coeur_rgb.shape[0], snow_x:snow_x+coeur_rgb.shape[1]]

        # Assure que coeur_mask et coeur_rgb ont la même taille
        if coeur_mask.shape != coeur_rgb.shape:
            min_height = min(coeur_mask.shape[0], coeur_rgb.shape[0])
            min_width = min(coeur_mask.shape[1], coeur_rgb.shape[1])
            coeur_mask = coeur_mask[:min_height, :min_width]
            coeur_rgb = coeur_rgb[:min_height, :min_width]

        # Masque pour éviter de dessiner le coeur sur le corps
        overlay[coeur_y:coeur_y+coeur_rgb.shape[0], snow_x:snow_x+coeur_rgb.shape[1]][~coeur_mask] = coeur_rgb[~coeur_mask]

    # Mélanger l'image du cadre avec les coeurs en arrière-plan
    result = cv2.addWeighted(frame, 1, overlay, 1, 0)
    # Mettre à jour le cadre
    frame[0:frame.shape[0],0:frame.shape[1]] = result
    

# Fonction pour mettre à jour l'affichage vidéo
def update_video():
    
    global panel,frame,filtreGlassesActiveB,filtreSepiaActiveB,filtreCoeursActiveB
    
    # Lire le flux vidéo de la webcam
    ret, frame = cap.read()
    
    if ret:
        
        # Convertir l'image de la webcam en format RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Appliquer le filtre seulement si l'utilisateur clic sur le bouton
        
        # Filtre pere noel
        if filtreGlassesActiveB:
            pereNoel(face_cascade,frame,img_chapeau,img_barbe)

        # Filtre sepia
        if filtreSepiaActiveB:
            sepia(frame)
        
        # Filtre coeurs
        if filtreCoeursActiveB:
            fondCoeurs(body_cascade,frame,img_coeur)
    

        # Convertir l'image en format PhotoImage pour pouvoir la mettre à jour dans le widget Label
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (640, 480))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)

        # Mettre à jour l'affichage
        panel.img = img
        panel.config(image=img)

    # Mettre à jour l'affichage vidéo après 10 ms
    root.after(10, update_video)
    
# Initialiser la webcam
cap = cv2.VideoCapture(0)

# Initialiser Tkinter
root = tk.Tk()
root.title("Application de Masques")


# Créer un widget Label pour afficher l'image
panel = tk.Label(root)
panel.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

# Interface utilisateur Tkinter
button_open_mask1 = tk.Button(root, text="Pere noel", command=filtreGlassesActive)
button_open_mask1.grid(row=1, column=0)

button_open_mask2 = tk.Button(root, text="Filtre", command=filtreSepiaActive)
button_open_mask2.grid(row=1, column=1)

button_open_mask2 = tk.Button(root, text="Fond coeurs", command=filtreCoeursActive)
button_open_mask2.grid(row=1, column=2)

# Mettre à jour l'affichage vidéo
update_video()

# Démarrer la boucle Tkinter
root.mainloop()


# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
