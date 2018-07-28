Collaborative Filtering(CF) based Recommendation Systems
======================

### Enviroment versions required:

Spark: 2.2.1  
Python: 2.7  
Scala: 2.11  

### Dataset used for testing:
[MovieLens](https://grouplens.org/datasets/movielens/) 

  
### 1\. Model-based Collaborative Filtering using Spark's [MLlib Collaborative Filtering - RDD-based API](http://spark.apache.org/docs/latest/mllib-collaborative-filtering.html) 
---------------------------------------

_Parameters used for ALS training:_  
_I used grid search to find the optimal parameters for training ALS Model for both Python and Scala. For doing this I used spark’s in-built ParamGridBuilder and RegressionValueEvaluator and received the following parameters to meet required RMSE value:_  
_1. rank – Number of features to use for training, I used 12 as rank._  
_2. iterations – Number of iterations of ALS, I used 12 iterations for training ALS._  
_3. lambda – Regularization parameter, I used 0.1 as lambda value.4. blocks – Number of blocks used to parallelize the computation, I used 1 block._  
_5. seed – Random seed for initial matrix factorization model, I used 4 as random seed._  

### Python command for executing Model-based Collaborative Filtering

* * *

Exceuting Model-based CF using _“Prasad\_Bhagwat\_ModelBasedCF.py”_

    spark-submit --driver-memory 4G --executor-memory 4G Prasad_Bhagwat_ModelBasedCF.py <ratings file path> <testing file path>
      
  
where,  
_<ratings file path>_ corresponds to the absolute path of input _‘ratings.csv’_ file  
_<testing file path>_ corresponds to the absolute path of input _‘testing.csv’_ file  

Example usage of the above command is as follows:

    ~/Desktop/spark-2.2.1/bin/spark-submit --driver-memory 4G Prasad_Bhagwat_task2_ModelBasedCF.py ratings_small.csv testing_small.csv 
    

Note : _Output file_ named _‘Prasad\_Bhagwat\_ModelBasedCF.txt’_ is generated at the location from where the _spark-submit_ is run.  
Note : Entries present in _‘testing.csv’_ file are not used to the your recommendation system.   
Note : If you get ’_Spark java.lang.OutOfMemoryError: Java heap space_ ’ this error while execution please change the _--executor-memory_ and _--driver-memory_ from _4G_ to _6G_ or _8G_

**ModelBasedCF Results using Python :**

RMSE for Small Movie Dataset: 0.95170869627   


### Scala command for executing Model-based Collaborative Filtering

* * *

Exceuting Model-based CF using _“Prasad\_Bhagwat\_RecommendationSystems.jar”_

    spark-submit --driver-memory 4G --executor-memory 4G --class ModelBasedCF Prasad_Bhagwat_RecommendationSystems.jar <ratings file path> <testing file path>
    

where,  
_<ratings file path>_ corresponds to the absolute path of input _‘ratings.csv’_ file  
_<testing file path>_ corresponds to the absolute path of input _‘testing.csv’_ file

Example usage of the above command is as follows:

    ~/Desktop/spark-2.2.1/bin/spark-submit --driver-memory 4G --class ModelBasedCF Prasad_Bhagwat_RecommenderSystems.jar ratings_small.csv testing_small.csv 
    

Note : _Output file_ named _‘Prasad\_Bhagwat\_ModelBasedCF.txt’_ is generated at the location from where the _spark-submit_ is run.  
Note : Entries present in _‘testing.csv’_ file are not used to the your recommendation system.   
Note : If you get ’_Spark java.lang.OutOfMemoryError: Java heap space_ ’ this error while execution please change the _–executor-memory_ and _–driver-memory_ from _4G_ to _6G_ or _8G_

**ModelBasedCF Results using Scala :**

RMSE for Small Movie Dataset: 0.9517086967596017   


### 2\. User-based Collaborative Filtering
-----------------

_Improvements done both in Python and Scala code to meet better RMSE:_  
_1. I have used Pearson Correlation as user-user similarity metric_  
_2. I have normalized the final user’s rating predictions which were going below 0 and above 5. Negative rating predictions were restricted to 0 and prediction value greater than 5 were restricted to 5 as the rating range was 0-5._  
_3. I have used average of user’s rating for missing data points_  

### Python command for executing User-based Collaborative Filtering

* * *

Exceuting User-based CF using _“Prasad\_Bhagwat\_task2_UserBasedCF.py”_

    spark-submit --driver-memory 4G Prasad_Bhagwat_task2_UserBasedCF.py <ratings file path> <testing file path>
    

where,  
_<ratings file path>_ corresponds to the absolute path of input _‘ratings.csv’_ file  
_<testing file path>_ corresponds to the absolute path of input _‘testing.csv’_ file

Example usage of the above command is as follows:

    ~/Desktop/spark-2.2.1/bin/spark-submit --driver-memory 4G Prasad_Bhagwat_task2_UserBasedCF.py ratings.csv testing_small.csv 
    

Note : _Output file_ named _‘Prasad\_Bhagwat\_UserBasedCF.txt’_ is generated at the location from where the _spark-submit_ is run.   
Note : Entries present in _‘testing.csv’_ file are not used to the your recommendation system.   

**UserBasedCF Results using Python :**

RMSE for Small Movie Dataset: 0.943994254061   


### Scala command for executing User-based Collaborative Filtering

* * *

Exceuting User-based CF using _“Prasad\_Bhagwat\_RecommendationSystems.jar”_

    spark-submit --driver-memory 4G --class UserBasedCF Prasad_Bhagwat_RecommendationSystems.jar <ratings file path> <testing file path>
    

where,  
_<ratings file path>_ corresponds to the absolute path of input _‘ratings.csv’_ file  
_<testing file path>_ corresponds to the absolute path of input _‘testing.csv’_ file

Example usage of the above command is as follows:

    ~/Desktop/spark-2.2.1/bin/spark-submit --driver-memory 4G --class UserBasedCF Prasad_Bhagwat_RecommenderSystems.jar ratings.csv testing_small.csv 
    

Note : _Output file_ named _‘Prasad\_Bhagwat\_UserBasedCF.txt’_ is generated at the location from where the _spark-submit_ is run.   
Note : Entries present in _‘testing.csv’_ file are not used to the your recommendation system.   

**UserBasedCF Results using Scala :**

RMSE for Small Movie Dataset: 0.9439942551274575   
   

### 3\. Item-based Collaborative Filtering using results obtained from results of [Jaccard Similarity](https://github.com/prasad-bhagwat/Locality-Sensitive-Hashing-using-Jaccard-Similarty)
-----------------

_Improvements done both in Python and Scala code to meet the required RMSE:_  
_1. To predict rating of movie for which Jaccard similarity is not present I have used average rating of the user for all other movies which are rated by that user in the training set which in turn increased the prediction accuracy._  

### Python command for executing Item-based Collaborative Filtering

* * *

Exceuting Item-based CF using _“Prasad\_Bhagwat\_task2_ItemBasedCF.py”_

    spark-submit --driver-memory 4G Prasad_Bhagwat_task2_ItemBasedCF.py <ratings file path> <testing file path> <movie similarity file path>
    

where,  
_<ratings file path>_ corresponds to the absolute path of input _‘ratings.csv’_ file  
_<testing file path>_ corresponds to the absolute path of input _‘testing.csv’_ file  
_<movie similarity file path>_ corresponds to the absolute path of input _‘movie_similarity.txt’_ file which is generated in advance using [Jaccard Similarity](https://github.com/prasad-bhagwat/Locality-Sensitive-Hashing-using-Jaccard-Similarty)

Example usage of the above command is as follows:

    ~/Desktop/spark-2.2.1/bin/spark-submit --driver-memory 4G Prasad_Bhagwat_task2_ItemBasedCF.py ratings.csv testing_small.csv 
    

Note : _Output file_ named _‘Prasad\_Bhagwat\_ItemBasedCF.txt’_ is generated at the location from where the _spark-submit_ is run.   
Note : Entries present in _‘testing.csv’_ file are not used to the your recommendation system.   

**ItemBasedCF Results using Python :**

RMSE for Small Movie Dataset: 0.982842925443


### Scala command for executing Item-based Collaborative Filtering

* * *

Exceuting Item-based CF using _“Prasad\_Bhagwat\_RecommendationSystems.jar”_

    spark-submit --driver-memory 4G --class ItemBasedCF Prasad_Bhagwat_RecommendationSystems.jar <ratings file path> <testing file path> <movie similarity file path>
    

where,  
_<ratings file path>_ corresponds to the absolute path of input _‘ratings.csv’_ file  
_<testing file path>_ corresponds to the absolute path of input _‘testing.csv’_ file  
_<movie similarity file path>_ corresponds to the absolute path of input _‘movie_similarity.txt’_ file which is generated in advance using [Jaccard Similarity](https://github.com/prasad-bhagwat/Locality-Sensitive-Hashing-using-Jaccard-Similarty)

Example usage of the above command is as follows:

    ~/Desktop/spark-2.2.1/bin/spark-submit --driver-memory 4G --class ItemBasedCF Prasad_Bhagwat_RecommenderSystems.jar ratings.csv testing_small.csv 
    

Note : _Output file_ named _‘Prasad\_Bhagwat\_ItemBasedCF.txt’_ is generated at the location from where the _spark-submit_ is run.   
Note : Entries present in _‘testing.csv’_ file are not used to the your recommendation system.   

**ItemBasedCF Results using Scala :**

RMSE for Small Movie Dataset: 0.9828429554399426
