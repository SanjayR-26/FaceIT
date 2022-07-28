from flask import Flask, render_template, url_for, request, redirect, Response, flash
from pymongo import MongoClient
from flask_sqlalchemy import SQLAlchemy
import face_recognition
import cv2
import asyncio
import numpy as np
import time
import os
from datetime import datetime
from flask_caching import Cache
from flask import send_file
cache = Cache(config={'CACHE_TYPE': 'simple'})

app= Flask(__name__)
app.secret_key = 'faceit'
cache.init_app(app)
dbauth = MongoClient("mongodb+srv://faceit:faceit@cluster0.zxqly.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
# APP_ROOT = os.path.dirname(os.path.abspath(__file__))
# UPLOAD_FOLD = 'database/images'
# UPLOAD_FOLDER = os.path.join(APP_ROOT, UPLOAD_FOLD)
app.config['UPLOAD_FOLDER'] = "database/images"




@cache.cached(timeout=3600, key_prefix='load_data')
def load_data():
    known_face_encodings =[]
    known_face_names=[]
    count = dbauth["Employee"]["Encodings"].count_documents({})
    for i in range(1,count+1):
        employee = dbauth["Employee"]["Encodings"].find_one({"_id": i})
        known_face_encodings.append(employee['Encodings'])
        known_face_names.append(employee['Name'])

    return known_face_encodings, known_face_names
       
def markAttendance(name):  
    with open('database/Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        now = datetime.now()
        date = now.strftime("%x")
        NameDate = f'{name}-{date}'
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if NameDate not in nameList:
            name = name
            Time = now.strftime('%X')
            f.writelines(f'{NameDate},{name},{date},{Time}\n')
    
def push(frame):
    now = datetime.now()
    dtString = now.strftime('%H.%M.%S')
    path = 'unknown'
    cv2.imwrite(f'{path}/Unknown{dtString}.jpg', frame)

async def createEmployeeindb(image, name):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(img)
    encodeList = face_recognition.face_encodings(img, facesCurFrame)[0]
    encodeList = encodeList.tolist()
    count = dbauth["Employee"]["Encodings"].count_documents({})
    dbauth["Employee"]["Encodings"].insert_one({"_id": count+1,"Name": name, "Encodings":encodeList })
    
def updateEmployeeindb(image, name):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(img)
    encodeList = face_recognition.face_encodings(img, facesCurFrame)[0]
    encodeList = encodeList.tolist()
    filter = {"Name": name}
    new = {"$set":{"Encodings":encodeList}}
    if dbauth["Employee"]["Encodings"].find_one(filter):
        dbauth["Employee"]["Encodings"].update_one(filter, new)
        return "Updated"
    return "Nouserfound"


def gen_frames():  
    cap = cv2.VideoCapture(0)
    while True:
            success, img = cap.read()
            imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)    
            
            if facesCurFrame:
                known_face_encodings,known_face_names = load_data()
                time.sleep(1)
                ...
                for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
                    faceDis = face_recognition.face_distance(known_face_encodings,encodeFace)
                    matchIndex = np.argmin(faceDis)
                    y1,x2,y2,x1 = faceLoc 
                    if faceDis[matchIndex]< 0.50:  
                        name = known_face_names[matchIndex].upper()
                        markAttendance(name)
                        # flash(f'Hello {name}! You are recorded')
                        cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                        cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
                    else:
                        push(img)
                        cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                        cv2.putText(img,"Thankyou",(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method =="POST":
        user = request.form["Username"]
        password = request.form["Password"]
        
        admin_data = dbauth["Employee"]["login"].find_one({"Username": user, "Password": password})
        if admin_data:
            return redirect(url_for("home"))
        else:
            return render_template('login.html')
    else: 
        return render_template('login.html')

@app.route('/home/')
def home():
    return render_template('home.html')

@app.route('/employee/')
def employee():
    return render_template('employee.html')    

@app.route('/error/')
def error():
    return render_template('error.html')

@app.route('/attendance')
def attendance():
    return render_template('attendance.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/download/')
def downloadFile ():
    path = "database/Attendance.csv"
    return send_file(path, as_attachment=True)

@app.route('/createEmployee',  methods=['POST'])
async def createEmployee():
    image = request.files["image"]
    filestr = request.files['image'].read()
    npimg = np.fromstring(filestr, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    image.save(image.filename)
    filename = image.filename.split(".")
    name = filename[0]
    await createEmployeeindb(img, name)
    return redirect(url_for('employee'))
    
@app.route('/updateEmployee',  methods=['POST'])
def updateEmployee():
    image = request.files["image"]
    filestr = request.files['image'].read()
    npimg = np.fromstring(filestr, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    image.save(image.filename)
    filename = image.filename.split(".")
    name = filename[0]
    if updateEmployeeindb(img, name) == "Updated":
        return redirect(url_for('employee'))
    elif updateEmployeeindb(img, name) == "Nouserfound":
        return redirect(url_for('error'))

if __name__ == "__main__":
    app.run(debug=True)