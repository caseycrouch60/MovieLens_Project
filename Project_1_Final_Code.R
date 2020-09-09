#Author: Casey Crouch, 9/1/2020
#HarvardX, Data Science: Capstone


#Project 1: MovieLens Recommendation System

#Goal: Develop an accurate film recommendation system with RMSE < 0.86490

#Warning: This script can take about 2-3 hours to execute


###Part 1: Preparing R

##########################################################
# Load key packages, install if necessary, boost memory
##########################################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(pixiedust)) install.packages("pixiedust", repos = "http://cran.us.r-project.org")
if(!require(zoo)) install.packages("zoo", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(lubridate)
library(pixiedust)
library(zoo)

#Optional: Increase memory limit for extensive calculations (fill in parentheses)

memory.limit()

#Warning: memory limit might need to be adjusted for machine RAM 



###Part 2: Data Generation & Train/Test Set Partition

##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

#As per project instructions, this code was copied directly from the course website

# Note: this process could take a couple of minutes

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)



###Part 3: Data Pre-Processing & Preparation

##########################################################
# Clean edx & validation sets, define RMSE function, etc.
##########################################################

#Our primary dataset, "edx", comprises 9,000,055 entries spread across 6 predictors 

nrow(edx)
ncol(edx)

#These predictors are: User ID, Movie ID, Movie Rating, Review Timestamp, Movie Title, and Movie Genres

colnames(edx)

#Here are the first 3 entries of edx:

dust(head(edx, n = 3)) %>%
  sprinkle_colnames(userId = 'User ID', 
                    movieId = 'Movie ID', 
                    rating = 'Rating', 
                    timestamp = 'Timestamp',
                    title = 'Title',
                    genres = 'Genres')

#Immediately, we notice several issues with the data:

#1): the timestamp is not readable by a human
#2): the year of film publication is attached to the title variable
#3): multiple genres are listed for each entry; this could hinder genre analysis

##Below, we take steps to address these problems 
#Note: the following operations are performed on both the edx & validation sets for continuity
#Furthermore: these operations only change the format in which existing data is presented for analysis--they do not alter the actual information contained therein

#Convert review timestamp to a year format and remove original variable

edx$review_date <- as_datetime(edx$timestamp)
edx$review_date <- year(edx$review_date)
edx <- subset(edx, select = -timestamp)

validation$review_date <- as_datetime(validation$timestamp)
validation$review_date <- year(validation$review_date)
validation <- subset(validation, select = -timestamp)

#Create new variable for publishing year 

pattern <- '\\)$'

publishing_dates <- sapply(edx$title, function(x){
  info <- str_split(x, "\\(")
  new_date <- info[[1]][length(info[[1]])]
  new_date_2 <- str_remove(new_date, pattern)
  as.numeric(new_date_2)
})
publishing_dates_2 <- as.vector(publishing_dates)
edx$publishing_date <- publishing_dates_2

publishing_dates_v <- sapply(validation$title, function(x){
  info <- str_split(x, "\\(")
  new_date <- info[[1]][length(info[[1]])]
  new_date_2 <- str_remove(new_date, pattern)
  as.numeric(new_date_2)
})
publishing_dates_2_v <- as.vector(publishing_dates_v)
validation$publishing_date <- publishing_dates_2_v

#Remove publishing year from title column

pattern <- ' \\(\\d{4}\\)$'

edx$title <- str_remove(edx$title, pattern)

validation$title <- str_remove(validation$title, pattern)

#Separate pipe-connected genres for each film
#This step elongates the data, as each genre is assigned a new row 

edx <- edx %>%
  separate_rows(genres, sep = '\\|')

validation <- validation %>%
  separate_rows(genres, sep = '\\|')

#Define RMSE function for model assessment 

RMSE <- function(true_ratings, predicted_ratings){ 
  sqrt(mean((true_ratings - predicted_ratings)^2))}

#Here are the first 3 entries of post-transformation edx:

dust(head(edx, n = 3)) %>%
  sprinkle_colnames(userId = 'User ID', 
                    movieId = 'Movie ID', 
                    rating = 'Rating', 
                    review_date = 'Review Date',
                    publishing_date = 'Publishing Date',
                    title = 'Title',
                    genres = 'Genres')

#You can see how the data looks much more organized, with separate, clean columns for both review and publishing dates, as well as a new row for each genre assigned to each film 
#It's always a good idea to clean messy data before conducting one's analysis, whether or not all of the cleaned variables are used in the final model or not 
#At this stage, we are ready to move onto our data exploration, where we will better acquaint ourselves with the dataset and search for trends to inform the modeling process



###Part 4: Data Exploration

##########################################################
# Pursue insights on data to inform modeling 
##########################################################

##Introductory Exploration

#We'll get started by reconsidering the dataset, now with the first 10 entries:

dust(head(edx, n = 10)) %>%
  sprinkle_colnames(userId = 'User ID', 
                    movieId = 'Movie ID', 
                    rating = 'Rating', 
                    review_date = 'Review Date',
                    publishing_date = 'Publishing Date',
                    title = 'Title',
                    genres = 'Genres')

#Post-transformation, the edx data consists of 23,371,423 rows with 7 columns, seen below:

nrow(edx)
ncol(edx)

#These 7 predictors now consist of: 'User ID', 'Movie Id', 'Rating', 'Review Date', 'Publishing Date', 'Title', and 'Genres'

#At this time, it's appropriate to consider the fundamental assumptions of rating systems, as these assumptions form the basis of the trends for which we are now searching
#These assumptions are threefold:

#1): It is reasonable to compare films
#2): Some films are perceived to be better than others, according to various market & user tastes
#3): These differences in perception can be captured in a star rating system that aggregates many individual opinions

#If these assumptions are true, then we expect to observe variability among our edx film ratings

#We can test this assumption by looking at the distribution of edx film ratings, generated below:

qplot(edx$rating, main = 'Film Rating Distribution Histogram', xlab = 'Ratings', ylab = 'Count', geom = 'histogram', fill = I('springgreen2'), col = I('black'), binwidth = 0.5)

#Additionally, here is the post-transformation count for each star rating, presented in descending order:

edx %>%
  group_by(rating) %>%
  summarize(total = n()) %>%
  arrange(desc(total)) %>%
  dust() %>%
  sprinkle_colnames(rating = 'Film Rating', total = 'Total')
  
#We see that 4-star ratings are most common, followed by 3-stars and 5-stars, respectively

#Here is a summary of the ratings, with the mean, median, etc.

summary(edx$rating)

#The mean rating is about 3.53 stars, which makes sense--that's a very average score 

#To further visualize the ratings variability, here is a boxplot of the distribution:
#Note: this boxplot describes a random sample of 10,000 edx observations, as R was unable to generate a boxplot on the full dataset without crashing, even in qplot()

set.seed(1, sample.kind = 'Rounding')
mini_edx_ratings <- sample(edx$rating, 10000, replace = TRUE)

qplot(mini_edx_ratings, main = 'Film Rating Distribution Boxplot', xlab = 'Ratings', geom = 'boxplot', fill = I('springgreen2'))

#The ratings data appears heavily skewed to the left, with the median equal to the 3rd quartile value of 4 stars
#This means that at least half of all films will have an average rating around 4 stars, and most films will have a "positive" rating of at least 3 stars
#But even with these trends, the large whiskers and outlier values imply that significant variability exists in the data as well

#Having observed such variability, and with the knowledge that some films tend to be more popular than others, we might expect to find that some films enjoy a disproportionate share of high ratings

#Here are the first 5 films with the highest mean rating (5-stars):

edx %>%
  group_by(title) %>%
  summarize(mean_rating = mean(rating)) %>%
  arrange(desc(mean_rating)) %>%
  slice(1:5) %>%
  dust() %>%
  sprinkle_colnames(title = 'Title', mean_rating = 'Average Rating')

#We might also expect to find that some films have a disproportionately negative share of ratings

#As such, here are the first 5 films with the lowest mean rating (0.5-stars):

edx %>%
  group_by(title) %>%
  summarize(mean_rating = mean(rating)) %>%
  arrange((mean_rating)) %>%
  slice(1:5) %>%
  dust() %>%
  sprinkle_colnames(title = 'Title', mean_rating = 'Average Rating')

#Given the existence of this general variability among the ratings, we can expect a "Movie Effect" to influence ratings
#This effect describes how the public views each film, on average, positively, negatively, or with moderate sentiments, all with respect to the average film rating

#Next, we turn to the influence of the user
#Knowing that the star ratings are given out by fellow humans, we can expect a level of personal bias to impact ratings

#Here is the distribution of ratings by User ID:

edx %>%
  group_by(userId) %>%
  summarize(mean_rating = mean(rating)) %>%
  ggplot(aes(userId, mean_rating)) +
  geom_col(aes(fill = mean_rating)) + 
  scale_fill_gradient(low = 'springgreen1', high = 'springgreen4') + 
  labs(x = 'User ID', y = 'Average Rating', fill = 'Average Rating') + 
  ggtitle('Average Rating by UserID + Rolling Mean') +
  geom_line(aes(y = rollmean(mean_rating, 1000, na.pad = T)), size = 1, color = 'black')

#Note: some columns are missing; this means that the user did not provide any reviews in the original data 

#And here is the distribution of the first 5 users in table form:

first_5_users <- unique(edx$userId)[1:5]
  
edx %>%
  filter(userId %in% first_5_users) %>%
  group_by(userId) %>%
  summarize(mean_rating = mean(rating)) %>%
  dust() %>%
  sprinkle_colnames(userId = 'UserID', mean_rating = 'Average Rating')
  
#We see that there is substantial variability among users for average ratings
#Some users appear to rate most movies very high, while others appear to rate most movies very low
#However, most users appear to favor moderate scores, ranging from about 3.5 to 4.5
#Altogether, this variability suggests the existence of a "User Effect", which describes how personal bias impacts ratings

#Finally, we turn to the influence of genres
#As we expect some movies to be more popular with the public than others, it is also reasonable to expect some genres to be more popular than others as well

#Here is the distribution of ratings across genre:

edx %>%
  group_by(genres) %>%
  summarize(mean_rating = mean(rating)) %>%
  ggplot(aes(genres, mean_rating)) +
  geom_col(aes(fill = mean_rating)) + 
  scale_fill_gradient(low = 'springgreen1', high = 'springgreen4') + 
  labs(x = 'Genre', y = 'Average Rating', fill = 'Average Rating') + 
  theme(axis.text.x = element_text(angle = 90)) + 
  ggtitle('Average Rating by Genre')

#And here is the distribution of the first 5 genres in table form:

first_5_genres <- edx$genres[1:5]

edx %>%
  filter(genres %in% first_5_genres) %>%
  group_by(genres) %>%
  summarize(mean_rating = mean(rating)) %>%
  dust() %>%
  sprinkle_colnames(genres = 'Genre', mean_rating = 'Average Rating')

#As with users, we recognize variability with regards to genre (but to a more limited extent)
#Nearly all genre rating means lie between 3.4 and 4.0, so while variability does exist, it is likely not very significant
#Despite this, we expect a minor "Genre Effect" to explain some variability in the ratings data

#Having identified potential sources of influence in a "Movie Effect", "User Effect", and "Genre Effect", we are now ready to begin modeling 



###Part 5: Model Generation & Analysis

##########################################################
# Model film ratings using various predictors 
##########################################################

#To begin, we will partition our edx training data into 1) a training set and 2) a test set
#We will reserve the validation set for our final test of the completed model

#Generate the partition into training/testing sets 

set.seed(1, sample.kind = 'Rounding')
testid <- createDataPartition(edx$rating, times = 1, p = 0.1, list = FALSE)
training <- edx %>% slice(-testid)
testing <- edx %>% slice(testid)

#We also need to add in some language to confirm that the MovieID, UserID, and Genre data that exist in the training set are also in the test set

testing <- testing %>%
  semi_join(training, by = 'movieId') %>%
  semi_join(training, by = 'userId') %>%
  semi_join(training, by = 'genres')

#Generate training RMSE result data frame; we will use this to store the RMSE after we run each intermediate model, as well as the final RMSE

rmse_results <- data.frame(Iteration = character(), RMSE = numeric())

##A Note on the Method:

#We create a series of models using the partitioned training/testing sets, slowly iterating as we add in each effect identified earlier
#After running the first four models (a baseline model + 3 iterated effect models), we will re-run the effect models using regularization; we will expand on this concept later
#After running all 7 baseline + intermediate models, we will select the most successful model to serve as our final model & answer to the prompt
#We will re-run this final model, replacing the training/testing sets with the edx and validation sets
#Following this calculation, we will report the final RMSE and move to the conclusion



###Baseline Model




##Model 1: Naive Prediction

#To get started, we develop a "naive model"
#Naive approaches are the simplest way to predict, utilizing a commonly-understood baseline figure to generate an estimate
#For example, a naive approach to predicting today's weather is to consider what happened yesterday, assuming that the weather doesn't change that much between days

#In this case, the rating average can serve as our baseline:

avg_rating <- mean(training$rating)

#Define the model

model_1 <- avg_rating

#Calculate and present the RMSE

model_1_rmse <- RMSE(testing$rating, model_1)

rmse_results[nrow(rmse_results) + 1,] <- c('Naive Prediction', round(model_1_rmse, digits = 6))

dust(rmse_results)

#We see that the naive approach gives us an RMSE of ~1.05
#Since we want to get to at least RMSE < 0.86490, we need to improve this model
#We can do this by introducing the effects we uncovered earlier



###Intermediate Models



##Model 2 (Single Effect): Predicting by "Movie Effect", + Rating Average

#Establish rating average baseline and 1 effect

avg_rating <- mean(training$rating)

movie_effect <- training %>%
  group_by(movieId) %>%
  summarize(movie_effect = mean(rating - avg_rating))

#Define & evaluate the model

model_2 <- testing %>%
  left_join(movie_effect, by = 'movieId') %>%
  mutate(prediction = avg_rating + movie_effect) %>%
  .$prediction

#Calculate and present the RMSE

model_2_rmse <- RMSE(testing$rating, model_2)

rmse_results[nrow(rmse_results) + 1,] <- c('Single Effect', round(model_2_rmse, digits = 6))

dust(rmse_results)



##Model 3 (Double Effect): Predicting by "User Effect", + Movie + Rating Average

#Establish rating average baseline and 2 effects

avg_rating <- mean(training$rating)

movie_effect <- training %>%
  group_by(movieId) %>%
  summarize(movie_effect = mean(rating - avg_rating))

user_effect <- training %>%
  left_join(movie_effect, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(user_effect = mean(rating - avg_rating - movie_effect))

#Define & evaluate the model

model_3 <- testing %>%
  left_join(movie_effect, by = 'movieId') %>%
  left_join(user_effect, by = 'userId') %>%
  mutate(prediction = avg_rating + movie_effect + user_effect) %>%
  .$prediction

#Calculate and present the RMSE

model_3_rmse <- RMSE(testing$rating, model_3)

rmse_results[nrow(rmse_results) + 1,] <- c('Double Effect', round(model_3_rmse, digits = 6))

dust(rmse_results)



##Model 4 (Triple Effect): Predicting by "Genre Effect", + User + Movie + Rating Average

#Establish rating average baseline and 3 effects

avg_rating <- mean(training$rating)

movie_effect <- training %>%
  group_by(movieId) %>%
  summarize(movie_effect = mean(rating - avg_rating))

user_effect <- training %>%
  left_join(movie_effect, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(user_effect = mean(rating - avg_rating - movie_effect))

genre_effect <- training %>%
  left_join(movie_effect, by = 'movieId') %>%
  left_join(user_effect, by = 'userId') %>%
  group_by(genres) %>%
  summarize(genre_effect = mean(rating - avg_rating - movie_effect - user_effect))

#Define & evaluate the model

model_4 <- testing %>%
  left_join(movie_effect, by = 'movieId') %>%
  left_join(user_effect, by = 'userId') %>%
  left_join(genre_effect, by = 'genres') %>%
  mutate(prediction = avg_rating + movie_effect + user_effect + genre_effect) %>%
  .$prediction

#Calculate and present the RMSE

model_4_rmse <- RMSE(testing$rating, model_4)

rmse_results[nrow(rmse_results) + 1,] <- c('Triple Effect', round(model_4_rmse, digits = 6))

dust(rmse_results)




##Further Improvements: Regularization

#By viewing our RMSE results so far, we can see that both the "Double Effect" and "Triple Effect" models have achieved an RMSE lower than the required value of 0.86490
#However, instead of moving to fit either model to the validation set, we can improve our model by applying regularization
#Regularization is an important concept in machine learning that describes how errors can be shrunk with the use of penalty terms
#Specifically, regularization penalizes large results that come out of small sample sizes; e.g. a film with only a few reviews having a perfect 5-star average
#We will re-run models 2-4 with regularization applied to each constituent effect and see if RMSEs continue to decrease



##Model 5 (Regularized Single): Regularized Single Effect Model, Movie + Rating Average

#Generate penalty terms for consideration

penalty <- seq(1, 10, 0.2)

#Create a data frame to store modeling results

penalty_results <- data.frame(penalty_term = numeric(), RMSE = numeric())

for(i in 1:length(penalty)){
  
  #Step 1: Establish rating average baseline and 1 effect  
  
  avg_rating <- mean(training$rating)
  
  movie_effect_regularized <- training %>%
    group_by(movieId) %>%
    summarize(movie_effect_regularized = sum(rating - avg_rating) / (n() + penalty[i]))
  
  #Step 2: Define & evaluate the model
  
  model_5_wip <- testing %>%
    left_join(movie_effect_regularized, by = 'movieId') %>%
    mutate(prediction = avg_rating + movie_effect_regularized) %>%
    .$prediction
  
  #Step 3: Calculate the RMSE
  
  model_5_wip_rmse <- RMSE(testing$rating, model_5_wip)
  
  #Step 4: Save results to data frame
  
  penalty_results[nrow(penalty_results) + 1,] <- c(i, model_5_wip_rmse)
}

#Identify the most efficient penalty term

best_penalty_model_5 <- penalty[which.min(penalty_results$RMSE)]

#Next, we plot RMSE vs Penalty Terms to visualize Model 5

ggplot(data = data.frame(penalty = penalty, RMSE = penalty_results$RMSE, min_RMSE = min(penalty_results$RMSE), min_penalty = penalty[which.min(penalty_results$RMSE)]), aes(penalty, RMSE)) +
  geom_point() +
  geom_point(aes(y = min_RMSE, x = min_penalty, color = 'red'), shape = 10, size = 7, stroke = 1.2) +
  theme(legend.position = 'none') + 
  labs(x = 'Penalty Term', y = 'RMSE', title = 'RMSE vs Penalty Terms: Regularized Single Model') 

#Save the minimum RMSE and present

model_5_rmse <- min(penalty_results$RMSE)

rmse_results[nrow(rmse_results) + 1,] <- c('Regularized Single', round(model_5_rmse, digits = 6))

dust(rmse_results)



##Model 6 (Regularized Double): Regularized Double Effect Model, User + Movie + Rating Average

#Generate penalty terms for consideration

penalty <- seq(1, 10, 0.2)

#Create a data frame to store modeling results

penalty_results <- data.frame(penalty_term = numeric(), RMSE = numeric())

for(i in 1:length(penalty)){
  
  #Step 1: Establish rating average baseline and 2 effects  
  
  avg_rating <- mean(training$rating)
  
  movie_effect_regularized <- training %>%
    group_by(movieId) %>%
    summarize(movie_effect_regularized = sum(rating - avg_rating) / (n() + penalty[i]))
  
  user_effect_regularized <- training %>%
    left_join(movie_effect_regularized, by = 'movieId') %>%
    group_by(userId) %>%
    summarize(user_effect_regularized = sum(rating - avg_rating - movie_effect_regularized) / (n() + penalty[i]))
  
  #Step 2: Define & evaluate the model
  
  model_6_wip <- testing %>%
    left_join(movie_effect_regularized, by = 'movieId') %>%
    left_join(user_effect_regularized, by = 'userId') %>%
    mutate(prediction = avg_rating + movie_effect_regularized + user_effect_regularized) %>%
    .$prediction
  
  #Step 3: Calculate the RMSE
  
  model_6_wip_rmse <- RMSE(testing$rating, model_6_wip)
  
  #Step 4: Save results to data frame
  
  penalty_results[nrow(penalty_results) + 1,] <- c(i, model_6_wip_rmse)
}

#Identify the most efficient penalty term

best_penalty_model_6 <- penalty[which.min(penalty_results$RMSE)]

#Next, we plot RMSE vs Penalty Terms to visualize Model 6 

ggplot(data = data.frame(penalty = penalty, RMSE = penalty_results$RMSE, min_RMSE = min(penalty_results$RMSE), min_penalty = penalty[which.min(penalty_results$RMSE)]), aes(penalty, RMSE)) +
  geom_point() +
  geom_point(aes(y = min_RMSE, x = min_penalty, color = 'red'), shape = 10, size = 7, stroke = 1.2) +
  theme(legend.position = 'none') + 
  labs(x = 'Penalty Term', y = 'RMSE', title = 'RMSE vs Penalty Terms: Regularized Double Model') 

#Save the minimum RMSE and present

model_6_rmse <- min(penalty_results$RMSE)

rmse_results[nrow(rmse_results) + 1,] <- c('Regularized Double', round(model_6_rmse, digits = 6))

dust(rmse_results)



#Increase memory limit to continue processing (local machine issue)
memory.limit(10000)



##Model 7 (Regularized Triple): Regularized Triple Effect Model, Genre + User + Movie + Rating Average

#Generate penalty terms for consideration

penalty <- seq(1, 10, 0.2)

#Create a data frame to store modeling results

penalty_results <- data.frame(penalty_term = numeric(), RMSE = numeric())

for(i in 1:length(penalty)){
  
  #Step 1: Establish rating average baseline and 3 effects  
  
  avg_rating <- mean(training$rating)
  
  movie_effect_regularized <- training %>%
    group_by(movieId) %>%
    summarize(movie_effect_regularized = sum(rating - avg_rating) / (n() + penalty[i]))
  
  user_effect_regularized <- training %>%
    left_join(movie_effect_regularized, by = 'movieId') %>%
    group_by(userId) %>%
    summarize(user_effect_regularized = sum(rating - avg_rating - movie_effect_regularized) / (n() + penalty[i]))
  
  genre_effect_regularized <- training %>%
    left_join(movie_effect_regularized, by = 'movieId') %>%
    left_join(user_effect_regularized, by = 'userId') %>%
    group_by(genres) %>%
    summarize(genre_effect_regularized = sum(rating - avg_rating - movie_effect_regularized - user_effect_regularized) / (n() + penalty[i]))
  
  #Step 2: Define & evaluate the model
  
  model_7_wip <- testing %>%
    left_join(movie_effect_regularized, by = 'movieId') %>%
    left_join(user_effect_regularized, by = 'userId') %>%
    left_join(genre_effect_regularized, by = 'genres') %>%
    mutate(prediction = avg_rating + movie_effect_regularized + user_effect_regularized + genre_effect_regularized) %>%
    .$prediction
  
  #Step 3: Calculate the RMSE
  
  model_7_wip_rmse <- RMSE(testing$rating, model_7_wip)
  
  #Step 4: Save results to data frame
  
  penalty_results[nrow(penalty_results) + 1,] <- c(i, model_7_wip_rmse)
}

#Identify the most efficient penalty term

best_penalty_model_7 <- penalty[which.min(penalty_results$RMSE)]

#Next, we plot RMSE vs Penalty Terms to visualize Model 7 

ggplot(data = data.frame(penalty = penalty, RMSE = penalty_results$RMSE, min_RMSE = min(penalty_results$RMSE), min_penalty = penalty[which.min(penalty_results$RMSE)]), aes(penalty, RMSE)) +
  geom_point() +
  geom_point(aes(y = min_RMSE, x = min_penalty, color = 'red'), shape = 10, size = 7, stroke = 1.2) +
  theme(legend.position = 'none') + 
  labs(x = 'Penalty Term', y = 'RMSE', title = 'RMSE vs Penalty Terms: Regularized Triple') 

#Save the minimum RMSE and present

model_7_rmse <- min(penalty_results$RMSE)

rmse_results[nrow(rmse_results) + 1,] <- c('Regularized Triple', round(model_7_rmse, digits = 6))

dust(rmse_results)



#Upon viewing our training results, we can see that the "Regularized Triple" model of Rating Average + Reg. Movie Effect + Reg. User Effect + Reg. Genre Effect obtained the lowest RMSE
#As a final step, we will run this model again with the validation set
#The RMSE generated by this model will serve as this project's final outcome



###Final Model



##Model 8: Final Model, (Regularized) Genre + User + Movie + Rating Average

#Step 1: Establish rating average baseline and 3 effects  

avg_rating <- mean(edx$rating)

movie_effect_regularized <- edx %>%
  group_by(movieId) %>%
  summarize(movie_effect_regularized = sum(rating - avg_rating) / (n() + best_penalty_model_7))

user_effect_regularized <- edx %>%
  left_join(movie_effect_regularized, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(user_effect_regularized = sum(rating - avg_rating - movie_effect_regularized) / (n() + best_penalty_model_7))

genre_effect_regularized <- edx %>%
  left_join(movie_effect_regularized, by = 'movieId') %>%
  left_join(user_effect_regularized, by = 'userId') %>%
  group_by(genres) %>%
  summarize(genre_effect_regularized = sum(rating - avg_rating - movie_effect_regularized - user_effect_regularized) / (n() + best_penalty_model_7))

#Step 2: Define & evaluate the model

model_8 <- validation %>%
  left_join(movie_effect_regularized, by = 'movieId') %>%
  left_join(user_effect_regularized, by = 'userId') %>%
  left_join(genre_effect_regularized, by = 'genres') %>%
  mutate(prediction = avg_rating + movie_effect_regularized + user_effect_regularized + genre_effect_regularized) %>%
  .$prediction

#Step 3: Save the final predicted ratings

final_ratings <- model_8

#Step 4: Calculate the RMSE and present

model_8_rmse <- RMSE(validation$rating, model_8)

rmse_results[nrow(rmse_results) + 1,] <- c('Final Model', round(model_8_rmse, digits = 6))

dust(rmse_results)

#Note: I switched from using the "training" data to the "edx" data for Model 8 because the partition caused the training data to lose some of the MovieID values present in "edx" that are needed to evaluate against the validation data 



#With a RMSE of 0.862927, our final model has cleared the RMSE < 0.86490 requirement, and we will move on to the conclusion



###Part 6: Summary of Results

##########################################################
# Outline results & look to the future
##########################################################

#In this project, we developed a film recommendation system
#First, we cleaned the data, then we looked for trends according to movie, user, and genre
#Finally, we integrated the effects deduced from the data exploration into a series of models, based around a training/testing partition of the overall training data, "edx"
#Our final model used regularization and three effects, along with a baseline value, to generate predictions that resulted in an RMSE < 0.86490

#Once again, here are the RMSE results of models 1-7 and the final model

dust(rmse_results)

#Additionally, here are the first 10 predicted ratings that were generated by our final model

head(final_ratings, n = 10)

#The full list of predicted movie ratings can be found in the "final_ratings" vector

##Potential Limitations
#1): This study could not take advantage of complex machine learning algorithms, such as kNN and Random Forests due to the size of the dataset and limited computing power
#2): Additionally, this study could not produce all of its plots using the full edx dataset, as R would repeatedly crash even with the memory limit increase
#3): This study focused on effects arising from movies, users, and genres, and it did not consider other potential influences on the ratings, such as time of release and time of review upload

##Suggestions for Future Research
#1): This study could be significantly improved with the use of machines with improved processing power such that automated machine learning methods would become viable
#2): Future researchers should conduct a more thorough exploration of the data to uncover trends beyond effects caused by movies, users, and genres
#3): In the era of COVID-19, more people are watching streaming services than ever, and it could be interesting to see if new trends emerge in the data; an interesting study would be comparing rating habits pre-quarantine and post-quarantine










