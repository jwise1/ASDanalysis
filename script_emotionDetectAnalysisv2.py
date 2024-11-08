import os
import pandas as pd
from pathlib import Path
from deepface   import DeepFace
import matplotlib.pyplot as plt
from PIL import Image








def predictBatch(input_dir, output_dir):


    
    i=0
    ids=[]
    files=[]
    results=[]
    results2=[]
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.jpg') or  file_name.endswith('.png') or file_name.endswith('.jpeg') or file_name.endswith('.JPG'):
            input_file_path = os.path.join(input_dir, file_name)
            
            result = DeepFace.analyze(img_path=input_file_path,actions=["emotion"],enforce_detection=False, detector_backend= 'mtcnn',align=True)
            result1=result[0]['emotion']
            result2=result[0]['dominant_emotion']
            output_dir2=output_dir+result2+"/"
            Path(output_dir2).mkdir(parents=True, exist_ok=True)
            img=Image.open(input_file_path)
            img.save(output_dir2+file_name)
            # csv file input
            uniqueID="img_"+str(i)
            ids.append(uniqueID)
            files.append(input_file_path)
            results.append(result1)
            results2.append(result2)
            i+=1
    Happy = []
    Sad = []
    Angry = []
    Fear = []
    Surprise = []
    Disgust = []
    Neutral =[]
    for i in results: 
        Happy.append(i['happy'])
        Sad.append(i['sad'])
        Angry.append(i['angry'])
        Fear.append(i['fear'])
        Surprise.append(i['surprise'])
        Disgust.append(i['disgust'])
        Neutral.append(i['neutral'])




        
    
    dataCSV=pd.DataFrame({'ID' : [a for a in ids], 'File Name' : [b for b in files], 'Happy' : [c for c in Happy], 'Sad' : [d for d in Sad] , 
                          'Angry' : [e for e in Angry], 'Fear' : [f for f in Fear],  'Surprise' : [g for g in Surprise], 'Disgust' : [h for h in Disgust], 'Neutral' : [i for i in Neutral],'Dominant Emotion' : [j for j in results2]}, columns=['ID', 'File Name', 'Happy', 'Sad', 'Angry', 'Fear', 'Surprise','Disgust','Neutral', 'Dominant Emotion'])
    dataCSV.to_csv(output_data_dir+'imageDataAnalysis.csv',index=False)


# Define input and output directories
input_data_dir = './data/Kaggle_BD'
output_data_dir = './KBD_Data_Analysis_retinaface_test/'


# Normalize CSV files in the input directory
predictBatch(input_data_dir, output_data_dir)


print("Analysis complete. File located in Data Analysis folder.")