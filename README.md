# ai-match-scene
can train from images of your kitchen then it can detect if the kettle is moved

Capture the training images from your laptop camera and send over network to train model on server.

Server also can "match" if the scene matches or not.


use captureAndPostTrainingImages.sh and detectAnomoly.sh on the remote laptop to feed the camera photos to the server

run TrainAndPredict.py on the server to train or detection backend processing.



TODO: 
  * requirements.txt
  * fix the train gphoto2 to use first available device - it think the detect already does this
