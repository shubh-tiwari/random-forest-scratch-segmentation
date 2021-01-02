# Scratch segmentation using random forest classifier
This repo contains the scripts to detect scratch defects on metal nut using random forest classifer

### Process flow chart
![Predicted sample](https://github.com/shubh-tiwari/random-forest-scratch-segmentation/blob/main/flow_chart/flowchart.png)

### Training random forest classifier
- Data acquisition
- Feature extraction using some of the imporatant filters and edge detection methods available in image processing
- Creating dataframe of these features and splitting training and validation dataset
- Training the random forest classifier
- Evaluating the accuracy score on validation dataset

#### The model gives the accuracy score of 95.39% on validation dataset

### Acknowledgment for the metal nut image dataset
Paul Bergmann, Michael Fauser, David Sattlegger and Carsten Steger, the authors of the [MVTec paper](#https://openaccess.thecvf.com/content_CVPR_2019/papers/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.pdf)
