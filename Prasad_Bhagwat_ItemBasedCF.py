# Imports required for the program
from pyspark import SparkConf, SparkContext
import time
import copy
import sys
import math


# Counting error buckets
def count_item_differences(actual, predicted):
    error = abs(actual - predicted)
    if error >= 0.0 and error < 1.0:
        return '>=0 and <1'
    elif error >= 1.0 and error < 2.0:
        return '>=1 and <2'
    elif error >= 2.0 and error < 3.0:
        return '>=2 and <3'
    elif error >= 3.0 and error < 4.0:
        return '>=3 and <4'
    else:
        return '>=4'


# Get nearest neighbors list in descending order based on jaccard similarities
def get_nearest_neighbors(user_movies_list, movie_1, jaccard_dict):
    nearest_neighbors = dict()
    for movie in user_movies_list:
        nearest_neighbors[movie] = jaccard_dict.get((movie_1, movie), 0.0)
    neighbors = sorted(nearest_neighbors, key=nearest_neighbors.get, reverse=True)
    return neighbors


# Calculating average of input list
def calculate_average(input_list):
   if len(input_list) > 0:
       return float(sum(input_list)) / len(input_list)
   else:
       return 0.0


# Main Function
def main():
    start_time = time.time()

    # Command Line Arguments
    input_file      = sys.argv[1]
    testing_file    = sys.argv[2]
    jaccard_file    = sys.argv[3]

    # Output filename generation
    output_file_name    = "Prasad_Bhagwat_ItemBasedCF.txt"
    output_file         = open(output_file_name, "w")

    # Creating Spark Context
    spark_config    = SparkConf()
    spark_context   = SparkContext(conf= spark_config)
    spark_context.setLogLevel("WARN")

    # Reading input training data file and extracting header
    input           = spark_context.textFile(input_file).filter(lambda x: "userId" not in x)
    input_data      = input.map(lambda x: ((int(x.split(",")[0]), int(x.split(",")[1])), float(x.split(",")[2])))

    # Reading test data file and extracting header
    testing         = spark_context.textFile(testing_file).filter(lambda x: "userId" not in x)
    testing_data    = testing.map(lambda x: ((int(x.split(",")[0]), int(x.split(",")[1])), None))

    # Reading jaccard similarity file
    jaccard_data    = spark_context.textFile(jaccard_file)
    jaccard_RDD     = jaccard_data.map(lambda x: ((int(x.split(", ")[0]), int(x.split(", ")[1])), float(x.split(", ")[2])))
    jaccard_dict    = dict(jaccard_RDD.collect())

    # Generating training data by subtracting testing data from it
    training_data   = input_data.subtractByKey(testing_data)

    # Creating RDD of tuples like ((user, movie), rating) and dictionary of (user, movie): rating
    usermovie_rating_RDD    = training_data.map(lambda x: ((x[0][0], x[0][1]), x[1]))
    usermovie_rating_dict   = dict(usermovie_rating_RDD.collect())

    # Creating RDD of tuples like (user, movie) and dictionary of user: set(movies)
    user_movie_RDD          = training_data.map(lambda x: (x[0][0], x[0][1])).groupByKey().mapValues(set)
    user_movie_dict         = dict(user_movie_RDD.collect())

    # Testing dictionary of user: movie
    testing_list            = testing.map(lambda x: (int(x.split(",")[0]), int(x.split(",")[1]))).collect()

    output_list             = list()

    # Generating Pearson co-relation among all co-rated users for each movie
    for tuple in testing_list:
        user                    = tuple[0]
        movie_1                 = tuple[1]
        # Get all users who rated Movie for which rating is to be predicted
        intermediate_user_movies= user_movie_dict.get(user, {0})
        user_movies_set         = copy.copy(intermediate_user_movies)
        user_rated_movies_list  = list()
        # Iterating over all the rated movies to generate list of the ratings given by User1
        for user_movie in user_movies_set:
            # Creating list of ratings of movies for User1
            user_rated_movies_list.append(usermovie_rating_dict.get((user, user_movie), 0.0))

        user_average     = calculate_average(user_rated_movies_list)

        # Remove movie1 from movie1_users
        user_movies_list = user_movies_set - {movie_1}
        user_movies      = get_nearest_neighbors(user_movies_list, movie_1, jaccard_dict)

        # Calculating average rating of User1 for all other movies
        numerator        = 0.0
        denominator      = 0.0

        for movie_2 in user_movies[:11]:
            if movie_1 < movie_2:
                #print usermovie_rating_dict.get((user, movie_2)), jaccard_dict.get((movie_1, movie_2), 3.5)
                numerator   += float(usermovie_rating_dict.get((user, movie_2), 0.0) * jaccard_dict.get((movie_1, movie_2), 0.0))
                denominator += float(abs(jaccard_dict.get((movie_1, movie_2), 0.0)))
            else:
                #print usermovie_rating_dict.get((user, movie_2)), jaccard_dict.get((movie_1, movie_2), 3.5)
                numerator   += float(usermovie_rating_dict.get((user, movie_2), 0.0) * jaccard_dict.get((movie_2, movie_1), 0.0))
                denominator += float(abs(jaccard_dict.get((movie_2, movie_1), 0.0)))

        # Predicting rating using values of Pearson co-relation if denominator is not 0
        if denominator != 0.0:
            predicted_rating = float(numerator / denominator)
            # If predicted rating is within range (1, 5) then keep as it is
            if predicted_rating > 0.0 and predicted_rating <= 5.0:
                output_list.append(((user, movie_1), predicted_rating))
            # If predicted rating is greater then 5 then truncate to 5
            elif predicted_rating > 5.0:
                output_list.append(((user, movie_1), 5.0))
            # If predicted rating is lesser then 0 then truncate to 0
            elif predicted_rating < 0.0:
                output_list.append(((user, movie_1), 0.0))
        # Predicting rating as average of user rating if denominator is 0
        else:
            output_list.append(((user, movie_1), user_average))

    # Generating output string as per expected format and writing in Output file
    output_result   = "\n".join(map(str, output_list)).replace("(", "").replace(")", "")
    output_file.write(output_result)
    output_file.close()

    # Joining actual ratings and predicted ratings for calculating RMSE
    output_rdd = spark_context.parallelize(output_list)
    real_predicted = input_data.map(lambda x: ((x[0][0], x[0][1]), x[1])).join(output_rdd)

    # Calculating and printing counts of error buckets
    differences = real_predicted.map(lambda x: count_item_differences(x[1][0], x[1][1])).countByValue()
    for key in sorted(differences.iterkeys()):
        print "%s: %s" % (key, differences[key])

    # Calculating MSE and RMSE values and printing RMSE
    MSE = float(real_predicted.map(lambda x: (x[1][0] - x[1][1]) ** 2).mean())
    RMSE = float(math.sqrt(MSE))
    print "RMSE: ", RMSE

    # Printing time taken by the program
    print "Time:", int(time.time() - start_time), "sec"

# Entry point of the program
if __name__ == '__main__':
    main()
