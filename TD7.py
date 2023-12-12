import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

def apply_sepia(frame):
    # Conversion en sépia
    sepia_filter = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])

    sepia_frame = cv2.transform(frame[0:frame.shape[0],0:frame.shape[1]], sepia_filter)

    frame[0:frame.shape[0],0:frame.shape[1]] = sepia_frame

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye_tree_eyeglasses.xml')

#glasses = cv2.imread('images/s.png', cv2.IMREAD_UNCHANGED)
img_chapeau = cv2.imread('images/chapeau.png', cv2.IMREAD_UNCHANGED)
img_barbe = cv2.imread('images/barbe.png', cv2.IMREAD_UNCHANGED)
frame = None
filtreGlassesActiveB = False
filtreSepiaActiveB = False

def filtreGlassesActive():
    global filtreGlassesActiveB
    filtreGlassesActiveB = not filtreGlassesActiveB
def filtreSepiaActive():
    global filtreSepiaActiveB
    filtreSepiaActiveB = not filtreSepiaActiveB
def filtreGlasses(face_cascade,eye_cascade,frame,glasses):
    # Convert into grayscale
    grayMultiple = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces
    facesMultiple = face_cascade.detectMultiScale(grayMultiple, 1.1, 4)
    if np.size(facesMultiple)>0:
        #Couper la video et recuperer la partie du visage
        for (x,y,w,h) in facesMultiple:
            frameEyes = frame[y:y+h, x:x+w]
            
            grayEyes =cv2.cvtColor(frameEyes,cv2.COLOR_BGR2GRAY)
            eyesMultiple = eye_cascade.detectMultiScale(grayEyes, 1.1, 4)
            if len(eyesMultiple) >= 2:
            # Sort the detected eyes by x-coordinate
                eyesMultiple = sorted(eyesMultiple, key=lambda x: x[0])
                # Get the leftmost and rightmost points
                x1 = eyesMultiple[0][0]
                x2 = eyesMultiple[1][0] + eyesMultiple[1][2]
                # Get the topmost and bottommost points
                y1 = min(eyesMultiple[0][1], eyesMultiple[1][1])
                y2 = max(eyesMultiple[0][1] + eyesMultiple[0][3], eyesMultiple[1][1] + eyesMultiple[1][3])
                # Calculate the width and height of the glasses
                w_glasses = x2 - x1
                h_glasses = y2 - y1
                # Resize the glasses and place them on the face
                glasses_resized = cv2.resize(glasses, (w_glasses, h_glasses))
                alpha_channel = glasses_resized[:, :, 3] / 255.0
                alpha_channel = np.expand_dims(alpha_channel, axis=2)
                result = alpha_channel * glasses_resized[:, :, :3] + (1.0 - alpha_channel) * frameEyes[y1:y2, x1:x2]
                frameEyes[y1:y2, x1:x2] = result.astype(np.uint8)
        frame[y:y + h, x:x + w] = frameEyes
    return frame

def chapeau(face_cascade,frame,img_chapeau,img_barbe) :
    
    
    # Convert into grayscale
    grayMultiple = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    facesMultiple = face_cascade.detectMultiScale(grayMultiple, 1.1, 4)

    if np.size(facesMultiple)>0:
        #Couper la video et recuperer la partie du visage
        for (x,y,w,h) in facesMultiple:
            w2=w
            h2=h
            hx1=x 
            hx2=hx1+w2
            hy1=y
            hy2=hy1+h2
                        
            img_chapeau_resized = cv2.resize(img_chapeau, (hx2-hx1,hy2-hy1))
            img_barbe_resized = cv2.resize(img_barbe, (hx2-hx1,hy2-hy1))
            
            chapeau_height, chapeau_width = img_chapeau_resized.shape[:2]
            barbe_height, barbe_width = img_barbe_resized.shape[:2]
            
            # Adjust y coordinate for the hat
            hy1_chapeau = hy1 - chapeau_height + 60
            
            # Adjust y coordinate for the hat
            hy1_barbe = hy1 + h - barbe_height + 100
            
            if hy1_chapeau >= 0:  # Ensure the hat is within the frame
                alpha_channel = img_chapeau_resized[:, :, 3] / 255.0
                alpha_channel = np.expand_dims(alpha_channel, axis=2)
                result = alpha_channel * img_chapeau_resized[:, :, :3] + (1.0 - alpha_channel) * frame[hy1_chapeau:hy1_chapeau+chapeau_height, hx1:hx1+chapeau_width]
                frame[hy1_chapeau:hy1_chapeau+chapeau_height, hx1:hx1+chapeau_width] = result.astype(np.uint8)
            
            if hy1_barbe >= 0:  # Ensure the hat is within the frame
                alpha_channel = img_barbe_resized[:, :, 3] / 255.0
                alpha_channel = np.expand_dims(alpha_channel, axis=2)
                result = alpha_channel * img_barbe_resized[:, :, :3] + (1.0 - alpha_channel) * frame[hy1_barbe:hy1_barbe + barbe_height, hx1:hx1 + barbe_width]
                frame[hy1_barbe:hy1_barbe+barbe_height, hx1:hx1+barbe_width] = result.astype(np.uint8)
    
# Fonction pour mettre à jour l'affichage vidéo
def update_video():
    global panel,frame,filtreGlassesActiveB,filtreSepiaActiveB
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # if filtreGlassesActiveB:
        #     filtreGlasses(face_cascade,eye_cascade,frame,glasses)
        if filtreGlassesActiveB:
            chapeau(face_cascade,frame,img_chapeau,img_barbe)
        # frame = apply_masks(frame)

        # Appliquer le filtre (sépia dans cet exemple)
        if filtreSepiaActiveB:
            apply_sepia(frame)
    

        # Convertir l'image en format PhotoImage
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (640, 480))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)

        # Mettre à jour l'affichage
        panel.img = img
        panel.config(image=img)

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

# Mettre à jour l'affichage vidéo
update_video()

# Démarrer la boucle Tkinter
root.mainloop()


# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
