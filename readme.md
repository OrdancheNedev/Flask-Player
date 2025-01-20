# Project Description


This web app is designed to analyze surveillance videos, specifically from abandoned buildings, to recognize people in them. The app integrates various technologies such as OpenCV, FaceNet, Flask, and Bootstrap, providing a seamless full-stack solution that demonstrates facial recognition, machine vision, and AI in action.

Built as a non-commercial research project by a college student, the goal is to showcase the potential of combining full-stack development with AI and machine vision in real-world applications. The app processes videos (usually 1-2 minutes long) to detect faces and identify people based on an uploaded image.



## Main page


![Alt text](https://raw.githubusercontent.com/OrdancheNedev/Flask-Player/master/image1.png)


![Alt text](https://raw.githubusercontent.com/OrdancheNedev/Flask-Player/master/image2.png)


## How It Works:

### Video and Image Upload:

     The user uploads a short surveillance video (1-2 minutes, typically from abandoned buildings) and an image of the person to be recognized.
     The app accepts the video and an image for facial recognition.

### Video Processing:

    Once uploaded, the app processes the video frame by frame using OpenCV. Each frame is analyzed for faces.
    The app compares the detected faces to the uploaded image using FaceNet. If a match is found, the frame is saved as an image in the upload/recognized folder.

### Download Recognized Frames:

    After analyzing the video, the app compiles the recognized frames into a zip file.
    The user can download the zip archive, containing all frames where the person from the uploaded image was detected.

### Automatic Cleanup:

    When a new video and image are uploaded, the server automatically deletes any previously uploaded video, image, recognized images, and the zip archive.
    This ensures the system remains efficient, only processing the latest files and preventing unnecessary data storage.


