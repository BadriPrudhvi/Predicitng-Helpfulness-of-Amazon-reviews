__author__ = 'Prudhvi Badri'


import gzip
import pandas as pd

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield eval(l)

output = parse("reviews_Movies_and_TV.json.gz")

# line = 0
# for reviews in output:
#     line+=1
# print line

reviewerID = []
productID = []
reviewerName = []
helpful = []
reviewText = []
ratings = []
reviewSummary = []
reviewTime = []
unixReviewTime = []

i = 0;
for review in output:
    # if(i<100000):
    reviewerID.append(review['reviewerID'])
    productID.append(review['asin'])
    if(not review.has_key('reviewerName')):
        ""
    elif(review['helpful'][0] > 10 and review['helpful'][0] < 8000):
        reviewerName.append(review['reviewerName'])
        helpful.append(review['helpful'])
        reviewText.append(review['reviewText'])
        ratings.append(review['overall'])
        reviewSummary.append(review['summary'])
        reviewTime.append(review['reviewTime'])
        unixReviewTime.append(review['unixReviewTime'])
    # else:
        # break
    i+=1
    print i

rID_series = pd.Series(reviewerID)
pID_series = pd.Series(productID)
rName_series = pd.Series(reviewerName)
help_series = pd.Series(helpful)
rText_series = pd.Series(reviewText)
rating_series = pd.Series(ratings)
rSummary_series = pd.Series(reviewSummary)
rTime_series = pd.Series(reviewTime)
uRTime_series = pd.Series(unixReviewTime)

df = pd.DataFrame([rID_series,pID_series,rName_series,help_series,rText_series,rating_series,rSummary_series,rTime_series,uRTime_series])
df1 = df.unstack().unstack()
df1.rename(columns={0: 'REVIEWER_ID', 1: 'PRODUCT_ID', 2 : 'REVIEWER_NAME', 3 :'HELPFUL', 4 : 'REVIEW_TEXT',5:'RATINGS',
                    6:'REVIEW_SUMMARY',7:'REVIEW_TIME', 8 : 'UNIX_REVIEW_TIME'}, inplace=True)
print df1.head(5)
df1.dropna(inplace=True)
print len(df1)
df1.to_csv("Movie_and_TV_reviews_data.csv")