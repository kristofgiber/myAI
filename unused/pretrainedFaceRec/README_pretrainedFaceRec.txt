
In the current version you can set boolean 'LOAD_PRETRAINED_MODEL' to True in the main activity.

It will then load 'pretrained_faces.xml' and the corresponding 'pretrained_faces_labels.txt' 
if they're present in the assets/pretrainedFaceRec/ folder.

If not used (LOAD_PRETRAINED_MODEL=false), delete 'pretrainedFaceRec/' folder 
to avoid compiling an unnecessarily big file in the apk.

NOTE: 
This strategy only makes sense with LBPH face recognizers!
Eigenface recognizers have no support for .update() method 
and thus you can only load a pretrained Eigenfaces model .xml file but not further train them by update.
Eigenface models can only be 'updated' if all the training images are loaded and 
re-trained from scratch each time. The latter method makes the app start about 5 seconds slower but on the
positive side, it takes about 50 times less memory space on the phone (see below) and is more stable.

PROS and CONS loading pretrained facerecognizer vs. raw training images:

PROS:
- app resume time is much faster: 8s for loading model vs. 12-13s for loading raw images (14s startup boot if app destroyed)
Note that I've tested it with a 27.3Mb model trained on 180 img, the booting could slow down even more if there are more images.
With raw images, however, the overall size is so small that the booting seems to be invariant to the number of images.

CONS:
- Saving the facerecognizer as .xml takes much longer and much more memory space than
saving the face images as jpg-s:
Eg. 180 JPGs of 80p x 80p take only 500Kb but 27.3Mb if 
stored as an input .xml file to re-create an Eigenfaces face recognizer.
- Because of the long file writing process with a limited phone processor, it also often
causes errors, or it might not properly finish saving the model when the app is closed etc.

