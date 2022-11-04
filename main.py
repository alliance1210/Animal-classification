from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import urllib.request
from PIL import Image
  
# urllib.request.urlretrieve(
#   'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSOIcIuJO11xsOXcTqbcdxTHlYvH0IjugwhwhdU5qL41W5QkcChJcK8ZnerzeJnnqlXwo4&usqp=CAU',
#    "gfg.png")
  
# img = Image.open("gfg.png")

# Load the model
model = load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
# Replace this with the path to your image
# image = Image.open('animals/animals/cats/cats_00001.jpg')
#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center
# image.show()



# print(model.predict(data))
# x=prediction.index(max(prediction))
ans=["Cat","Dog","Whale","Human","Bear","Chicken","Cow","Crocodile","Elephant","Goose","Hippopotamas","Horse",
"Lion","Lioness","Pig","Rhino","Sheep","Tiger","Rabbit"]

details=[]

details.append({"Height":"9 in","Weight":"4-5kg",
"Habitat":"The domestic cat is a cosmopolitan species and occurs across \nmuch of the world. It is adaptable and now present on all continents \nexcept Antarctica. Due to its ability to thrive in almost any \nterrestrial habitat, it is among the world's most invasive species. \nFeral cats can live in forests, grasslands, tundra, coastal areas, \nagricultural land, scrublands, urban areas, and wetlands.",
"Disease":"Diseases affecting domestic cats include acute \ninfections, parasitic infestations, injuries, and chronic diseases such as \nkidney disease, thyroid disease, and arthritis."})
details.append({"Height":"8-34in","Weight":"7-70kg",
"Habitat":"Dogs are domesticated animals that generally live in the same \nhabitats as humans.In the wild, dogs succeed in habitats that provide \nample food, water and cover, like forests and brush lands. For shelter, \nsome dogs will dig burrows, but most of the time they will use manmade \ncover or inhabit abandoned fox or coyote dens.",
"Disease":"Diseases affecting dogs include canine distemper, \nCanine influenza, canine parvovirus, external parasites (ticks, fleas and \nmange), fungal infections (blastomycosis, histoplasmosis, cryptococcosis, \ncoccidioidomycosis, etc.)"
})
details.append({"Height":"80ft","Weight":"136,078kg",
"Habitat":"Unlike fresh water dolphins whales live solely in saltwater \nenvironments, which is believed to have certain health properties that \nallow whales to heal from injuries quickly and avoid getting sick. Salt water \nenvironments also provide whales with the abundant food sources they \nneed in order to survive.",
"Disease":"Diseases affecting whales include calicivirus, \nequine encephalitides (WEE, EEE, VEE), protozoa, whale lice and \nbarnacles, biotoxins"
})
details.append({"Height":"1ft","Weight":"1kg"})
details.append({"Height":"1ft","Weight":"1kg"})
details.append({"Height":"1ft","Weight":"1kg"})
details.append({"Height":"1ft","Weight":"1kg"})
details.append({"Height":"1ft","Weight":"1kg"})
details.append({"Height":"9ft","Weight":"3000kg",
"Habitat":"They are found most often in savannas, grasslands, and \nforests but occupy a wide range of habitats, including deserts, swamps, \nand highlands in tropical and subtropical regions of Africa and Asia.",
"Disease":"Hemorrhage, Endoparasites, Gastrointestinal \nstasis or torsion, Lung lesions, Liver lesions"
})
details.append({"Height":"1ft","Weight":"1kg"})
details.append({"Height":"1ft","Weight":"1kg"})
details.append({"Height":"1ft","Weight":"1kg"})
details.append({"Height":"1ft","Weight":"1kg"})
details.append({"Height":"1ft","Weight":"1kg"})
details.append({"Height":"1ft","Weight":"1kg"})
details.append({"Height":"1ft","Weight":"1kg"})
details.append({"Height":"1ft","Weight":"1kg"})
details.append({"Height":"1ft","Weight":"1kg"})
details.append({"Height":"1ft","Weight":"1kg"})


# print(ans[x],":",round(prediction[x],2)*100,"%")
# print("Avg Height:",details[x]['Height'],", Avg Weight:",details[x]['Weight'])
# print("Habitat:",details[x]['Habitat'])
# print("Common health issues:",details[x]['Disease'])




import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
from tkinter import *

win = tk.Tk()
win.geometry("1920x1080")
win['background'] = '#829CD0'

font = ('georgia', 15, 'bold')
title = Label(win, text='AUTOMATED ANIMAL IDENTIFICATION AND DETECTION OF SPECIES')
title.config(bg='#20368F', fg='white')
title.config(font=font)
title.config(height=4, width=110)
title.place(x=0, y=10)

font1 = ('times', 12, 'bold')
text = Text(win, height=20, width=70)

# myscrollbar=Scrollbar(text) # parameter textbox
# text.configure(yscrollcommand=myscrollbar.set)
# text.place(x=50, y=120)
# text.config(font=font1)

frame1 = Canvas(win, width= 550, height= 385, bg="white")
mytext=frame1.create_text(5, 30,anchor= tk.NW, text="", fill="black", font=('Helvetica 13 '))
mytext2=frame1.create_text(5, 55,anchor= tk.NW, text="", fill="black", font=('Helvetica 13 '))
mytext3=frame1.create_text(5, 105,anchor= tk.NW, text="", fill="black", font=('Helvetica 13 '))
mytext4=frame1.create_text(5, 240,anchor= tk.NW, text="", fill="black", font=('Helvetica 13 '))

frame1.pack()
frame1.place(x=50, y=120)

# font = ('black', 11, 'bold') #image title
# Imagetext = Label(win, text='IMAGE DISPLAY')
# Imagetext.config(bg='white', fg='dark goldenrod')
# Imagetext.config(font=font)
# Imagetext.config(height=3, width=20)
# Imagetext.place(x=850, y=530)

font = ('black', 12, 'bold') # prediction title
Imagetext2 = Label(win, text='**** PREDICTION ****')
Imagetext2.config(bg='white', fg='dark goldenrod')
Imagetext2.config(font=font)
Imagetext2.config(height=3, width=20)
Imagetext2.place(x=1220, y=120)
# Imagetext2.pack()

# font1 = ('times',15, 'bold') #predicted img
# predict = Text(win, height=3, width=20)
# predict.place(x=1220, y=200)
# predict.config(font=font1)


frame = Frame(win, width=500, height=385, bg="grey", colormap="new") #image box
frame.pack()
frame.place(x=650, y=120)

from PIL import Image
import matplotlib.image as mpimg




def upload_file():
    global img
    for widget in frame.winfo_children():
        widget.destroy()
    f_types = [('Jpg Files', '*.jpg')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    img=Image.open(filename)
    global imagetext2
    imagetext2=img
    # get_probabilities(img)
    img_resized=img.resize((500,400)) # new width & height
    img=ImageTk.PhotoImage(img_resized)
    label = Label(frame, image = img)
    label.pack()

def predict_output():
  for widget in frame1.winfo_children():
        widget.destroy()
  size = (224, 224)
  image = ImageOps.fit(imagetext2, size, Image.ANTIALIAS)

  #turn the image into a numpy array
  image_array = np.asarray(image)
  # Normalize the image
  normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
  # Load the image into the array
  data[0] = normalized_image_array

  # run the inference
  prediction = model.predict(data)[0].tolist()
  x=prediction.index(max(prediction))
  score=ans[x]+":",round(prediction[x],2)*100,"%"
  comon="Avg Height: "+details[x]['Height']+"\nAvg Weight:"+details[x]['Weight']
  habitat="Habitat: "+details[x]['Habitat']
  issues= "Common health issues: "+details[x]['Disease']

  frame1.itemconfig(mytext,text=score)
  frame1.itemconfig(mytext2,text=comon)
  frame1.itemconfig(mytext3,text=habitat)
  frame1.itemconfig(mytext4,text=issues)
  



def close():
   win.destroy()

b1 = tk.Button(win,text='Upload \n Photo', width=20,command = lambda:upload_file()) #upload button
b1.config(font=('times', 12, 'bold'))
b1.place(x=850, y=600)

predict = tk.Button(win,text='Predict', width=20,command = lambda:predict_output()) #upload button
predict.config(font=('times', 12, 'bold'))
predict.place(x=1230, y=200)

font = ('black', 10, 'bold')
probabilities = Label(win, text='PROBABILITIES OF EACH CLASS') #probability tag
probabilities.config(bg='white', fg='dark goldenrod')
probabilities.config(font=('times', 12, 'bold'))
probabilities.config(height=3, width=30)
probabilities.place(x=200, y=580)

exitButton = Button(win, text="Exit", command=close)
exitButton.place(x=1230, y=550)
exitButton.config(font=('times', 12, 'bold'), height=2,width=20)

win.mainloop()  # Keep the window open
