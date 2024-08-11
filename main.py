import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import streamlit as ss
from PIL import Image
import io
import numpy as np
import time


resnet_model = tf.keras.models.load_model('./ResNet_Model/best_ResNet50_model.h5')

class_labels = ["Dyed Lifted Polyps", "Dyed Resection Margins", "Esophagitis", "Normal Cecum", "Normal Pylorus", 
                "Normal Z Line", "Polyps", "Ulcerative Colitis"]


def predict(model, x):
    result = model.predict(x)
    return result


#------streamlit starts here----------------
def predict_image(uploaded_img):
    img = Image.open(io.BytesIO(uploaded_img.read()))
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    prediction = resnet_model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_class, confidence

def main():
    ss.title("Gastro-Intestinal Disease Prediction")
    ss.markdown("##")
    ss.subheader("Upload the image below....")
    input_image = ss.file_uploader(label="Upload image", type=['jpg', 'png'])

    if ss.button("Submit"):
        if image is not None:
            prediction, confidence = predict_image(input_image)
            prediction = prediction.upper()
            confidence = confidence * 100
            with ss.spinner(text="In progress"):
                time.sleep(120)
                ss.success("Image Uploaded Successfully!!")
            if confidence>30:
                ss.write("There is a", round(confidence,1)-1,"% chance for ", prediction)
            else:
                ss.write("The model's confidence is below 30%. It may indicate that there is no disease or the image shows a type of disease not included in the trained categories, or an irrelevant image has been uploaded.")
                
        else:
            ss.write("Make sure you image is in JPG/PNG Format.")
    ss.markdown("##")
    ss.header("Introduction")
    #To display any image...
    ss.image("First_img.png")
    ss.markdown("Endoscopy is a vital procedure for detecting gastrointestinal diseases, including polyps, cancer, infections, and more. Its use has increased significantly due to rising gastrointestinal issues, but it can be time-consuming and burdensome for patients. Deep learning offers a solution by enabling non-invasive disease detection, minimizing patient discomfort and improving efficiency.")
    ss.markdown('''***Dyed Resection Margin and Dyed Lifted Polyps:*** These appear as abnormal growths or protrusions with colored margins, often indicating potential areas for further examination or treatment.\n
***Esophagitis:*** Inflammation or irritation of the esophagus, often visible as redness or swelling in the lining of the esophagus.\n
***Normal Cecum, Pylorus, and Z Line:*** These appear as healthy, typical anatomical structures without any signs of abnormality or disease.\n
***Polyps:*** These are visible as abnormal growths or protrusions from the mucous membrane, which may vary in size and shape.\n
***Ulcerative Colitis:*** This appears as inflammation and ulcers in the lining of the colon and rectum, often visible during endoscopy as redness, swelling, and ulcerations. ''')
    

if __name__=='__main__':
    main()
        