from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import urllib.request
from PIL import Image
  
urllib.request.urlretrieve(
  'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSOIcIuJO11xsOXcTqbcdxTHlYvH0IjugwhwhdU5qL41W5QkcChJcK8ZnerzeJnnqlXwo4&usqp=CAU',
   "gfg.png")
  
# img = Image.open("gfg.png")

# Load the model
model = load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
# Replace this with the path to your image
image = Image.open('gfg.png')
#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center
# image.show()
size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)

#turn the image into a numpy array
image_array = np.asarray(image)
# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
# Load the image into the array
data[0] = normalized_image_array

# run the inference
prediction = model.predict(data)[0].tolist()
x=prediction.index(max(prediction))
ans=["Cat","Dog","Whale","Human","Bear","Chicken","Cow","Crocodile","Elephant","Goose","Hippopotamas","Horse",
"Lion","Lioness","Pig","Rhino","Sheep","Tiger","Rabbit"]

details=[]

details.append({"Height":"1ft","Weight":"1kg"})
details.append({"Height":"1ft","Weight":"1kg"})
details.append({"Height":"1ft","Weight":"1kg"})
details.append({"Height":"1ft","Weight":"1kg"})
details.append({"Height":"1ft","Weight":"1kg"})
details.append({"Height":"1ft","Weight":"1kg"})
details.append({"Height":"1ft","Weight":"1kg"})
details.append({"Height":"1ft","Weight":"1kg"})
details.append({"Height":"9ft","Weight":"3000kg",
"Habitat":"They are found most often in savannas, grasslands, and forests but occupy a wide range of habitats, including deserts, swamps, and highlands in tropical and subtropical regions of Africa and Asia.",
"Disease":"Hemorrhage, Endoparasites, Gastrointestinal stasis or torsion, Lung lesions, Liver lesions"
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


print(ans[x],":",round(prediction[x],2)*100,"%")
print("Avg Height:",details[x]['Height'],", Avg Weight:",details[x]['Weight'])
print("Habitat:",details[x]['Habitat'])
print("Common health issues:",details[x]['Disease'])