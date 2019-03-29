# Cloud Services PMU classifier

This is created to apply dmachine learning (ML) algorithms on PMU.
This will implement ML algorithms as micro service structure on cloud platform.
Models are created in Tensorflow and keras framework. 

The PMU have following data points:
 <ul style="list-style-type:disc">
  <li>Frequency</li>

 ![Data point](images/Frequency.png) <br>
 
_**Fig. 1:** Data points_
  <li>V RMS</li>

  ![Data point](images/V_RMS.png) <br>
  
_**Fig. 2:** Data points_
  <li>V phase angle</li>

  ![Data point](images/V_ph.angle.png) <br>
  
_**Fig. 3:** Data points_
  <li>I RMS</li>

 ![Data point](images/I_RMS.png) <br>
 
_**Fig. 4:** Data points_
  <li>I phase angle</li>

 </ul>
 
 
# Results

Confusion matrix

![Confusion matrix](images/confusion_matrix.png) <br>
_**Fig. 5:** Confusion matrix_
 
Accuracy
![Accuracy](images/Accuracy.png) <br>
_**Fig. 6:** Accuracy_
