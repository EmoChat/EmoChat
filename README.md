# EmoChat
This is an innovative chat app which enable effective communication with Emojis. It effectively maps face of the person to the right emoji, which is sent to the other person and is rendered in place of the actual face. This enables the users to have an experience of emotion accompanied chat without having to show the actual face.  This feature will make chatting with strangers easier and secure.

## Screenshot

![](https://github.com/EmoChat/EmoChat/blob/master/Outputs/screenshot.jpg)

## Quick Start

Please read and follow these instructions carefully to run the application:

1. Download the *apk* file in *Outputs* folder

2. Run it on an android device and provide the Camera permission when asked

3. During startup if it prompts to install OpenCV Loader accept and do the same

4. When the main screen of the app opens up you can find three main components of the app, namely, the camera image, Emotion Prediction of the other chat, and a toggle switch on the op right corner

5. The toggle switch is used to decide the chronological order of the chats. So if you are chatting with your friend using this app, make sure you choose a different position than each other.

   For eg. If A chooses off position then B has to choose on and vice-versa.

6. The chat is a dummy image as the main purpose of this project is to introduce the innovative idea of Emotion Accompanied chatroom.

7. Now enjoy a live video-like chat without having to compromise your privacy and security!  

8.  Feel free to send a PR that implements the chat section

#### For curious developers:

1. The app uses Firebase for sharing the emotion. So make sure you create a Firebase project and link it with his project. Also create and link a Real time Database with the following structure:

   ```
   root
   	|
   	|
   	emotion
               |
               1 :
               |
               2 :
   ```

2. Install and setup all the dependencies 



## Dependencies 

1. OpenCV Android Library - https://opencv.org/android/

2. Tensorflow Android Library - https://www.tensorflow.org/lite/guide/android

3. Firebase dependencies - https://firebase.google.com/docs/android/setup

   

## Creators

https://github.com/swamitagupta

https://github.com/akri16

https://github.com/agnivabasak

https://github.com/r-ush
