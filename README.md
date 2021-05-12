# Exam2

1. start TF lite gesture model
Use RPC loop to call TF lite gesture model, type "/Gesture_UI/run" on screen, then LED1 will light up and now it's in TF lite gesture model.

2. identify the gesture
Use machine learning result and accelerator values to identify the gesture, and the gesture ID will show on uLCD. And the gesture form will also show on screen.

3. identify the extract features
When identifing the gesture, we also collecting accelerator values with another function 'find_feature()', the feature is 'go down fast is 1' another is 0.
