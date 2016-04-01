require(xgboost)
require(methods)
require(DiagrammeR)
require(caret)
require(lattice)
require(ggplot2)
require(e1071)
require(Rtsne)
require(stats)
require(knitr)
require(corrplot)

weka_data_Set <- read.csv("C:/amazon_project/Electronics_Output/Text_Features.csv",header = T,sep =',')[1:24000,]
divide_data <- sample(2, nrow(weka_data_Set),replace = TRUE,prob = c(0.8,0.2))
train <- weka_data_Set[divide_data==1,]
test <- weka_data_Set[divide_data==2,]


train[c("REVIEW_TEXT")] <- list(NULL)
test[c("REVIEW_TEXT")] <- list(NULL)

train_data = train[,c(1:6,8:107)]
train_class = train[,c("CLASS")]

test_data = test[,c(1:6,8:107)]
test_class = test[,c("CLASS")]


train_x = data.matrix(train_data)
train_y = data.matrix(train_class)

test_x = data.matrix(test_data)
test_y = data.matrix(test_class)

# set.seed(1024)
weka_data_Set[c("REVIEW_TEXT")] <- list(NULL)
new_train_data <- weka_data_Set[,c(1:6,8:107)]
new_train_label <- weka_data_Set[,c("CLASS")]
col3 <- colorRampPalette(c("red","green","pink", "white","yellow","orange", "blue"))
corrplot(cor(new_train_data), method="color",tl.cex=1.2,tl.offset = 0.5,
         order = "alphabet",col = col3(20))

new_train_x <- data.matrix(new_train_data)
new_train_y <- data.matrix(new_train_label)

cv.res <- xgb.cv(data = new_train_x, nfold = 10 , label = new_train_y ,nround = 1000,
                 verbose = TRUE, eta = 0.03,nthread=100,
                 objective = "binary:logistic",eval_metric ='auc', prediction=TRUE)

min_auc_idx = which.max(cv.res$dt[, test.auc.mean])
print(cv.res$dt[min_auc_idx,])

# get CV's prediction decoding
prediction <- as.numeric(cv.res$pred>0.5)
print ("XGBOOST CLASSIFIER")
print(confusionMatrix(factor(new_train_label),factor(prediction)))

bst <- xgboost(data = train_x , label = train_y, max.depth = 10, verbose = FALSE,
               eta = 0.01, nthread = 100 , nround = min_auc_idx, objective = "binary:logistic",task="pred")
pred <- predict(bst, test_x)  
prediction_1 <- as.numeric(pred>0.5)
print(confusionMatrix(factor(test_y),factor(prediction_1)))

names = dimnames(new_train_x)[[2]]
importance_matrix = xgb.importance(names, model=bst)
gp = xgb.plot.importance(importance_matrix)
print(gp)
