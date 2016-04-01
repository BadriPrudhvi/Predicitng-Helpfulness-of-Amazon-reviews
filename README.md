# Predicitng-Helpfulness-of-Amazon-reviews

Amazon review data is available here : http://jmcauley.ucsd.edu/data/amazon/

After you get the review dataset:

1. Run the reading_review_data.py file to read the GZ file and create a csv file containing the review data which will be used for the analysis.
2. Next run the Features_Extraction.py file to extract text features such as TF_IDF and Latent semantic analysis features which will be given as an input to out classifier
3. Finally run the Classifier.py which takes the Text_Features.csv generated in step-2 to perform the classification using Random Forest and Gradient Boosting classifers.

Note : Extreme gradient boosting classifier code is also included, using the XG_Boost_Classifier.r you can use xg_boost classifier.
