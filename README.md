# Group Cohesion Prediction

### ***2nd*** place in EmotiW 2019 Group Cohesion Prediction Challenge
ICMI2019: Exploring Regularizations with Face, Body and Image Cues for Group Cohesion Prediction

# <center>Exploring Regularizations with Face, Body and Image Cues for Group Cohesion Prediction</center>
                  Da Guo, Kai Wang, Jianfei Yang, Kaipeng Zhang, Xiaojiang Peng and Yu Qiao
                   Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences
                            {da.guo, kai.wang, xj.peng, yu.qiao}@siat.ac.cn         

![image](https://github.com/DaleAG/Group_Cohesion_Prediction/blob/master/pipeline.png)
The system pipeline of our approach. It contains three kinds of CNN, namely image-based CNN, body-based CNN and face-based CNN. Particularly, we use cascade attention network structure to train our body-based CNN and face-based CNN. The final prediction is made by averaging all the scores of CNNs from three visual cues.

![image](https://github.com/DaleAG/Group_Cohesion_Prediction/blob/master/GroupCohesion_hourglass.png)
Illustration of our hourglass loss. The loss value becomes larger when the predicted GCS is near the middle of two adjacent levels.

# Requirement
+ pytorch
+ python >= 3.6

# Code 
We release the key code in our approach, including rank loss and hourglass loss. You can apply it to your own approach.
