import faceRecognition as fr

faces,faceID=fr.label_for_training_data("dataset")
face_recognizer=fr.train_classifier(faces,faceID)