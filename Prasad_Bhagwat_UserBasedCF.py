# Imports required for the program
from pyspark import SparkConf, SparkContext
import time
import copy
import sys
import math


# Counting error buckets
def count_item_differences(actual, predicted):
    error = abs(actual - predicted)
    if error >= 0 and error < 1:
        return '>=0 and <1'
    elif error >= 1 and error < 2:
        return '>=1 and <2'
    elif error >= 2 and error < 3:
        return '>=2 and <3'
    elif error >= 3 and error < 4:
        return '>=3 and <4'
    else:
        return '>=4'


# Calculating average of input list
def calculate_average(input_list):
   if len(input_list) > 0:
       return float(sum(input_list)) / len(input_list)
   else:
       return 0


# Pearson co-relation calculation
def calculate_pearson_corelation(list_1, list_2):
    # Calculating average of both lists
    list_1_avg      = calculate_average(list_1)
    list_2_avg      = calculate_average(list_2)
    numerator       = 0
    list_1_diff_sum = 0
    list_2_diff_sum = 0
    # Calculating numerator for Pearson co-relation
    for item in range(len(list_1)):
        list_1_diff     = list_1[item] - list_1_avg
        list_2_diff     = list_2[item] - list_2_avg
        numerator       += list_1_diff * list_2_diff
        list_1_diff_sum += list_1_diff ** 2
        list_2_diff_sum += list_2_diff ** 2
    # Calculating denominator for Pearson co-relation
    denominator = list_1_diff_sum * list_2_diff_sum

    # Returning Pearson Co-relation
    if denominator != 0:
        return numerator / math.sqrt(denominator)
    else:
        return 0


# Main Function
def main():
    start_time = time.time()

    # Command Line Arguments
    input_file      = sys.argv[1]
    testing_file    = sys.argv[2]

    # Output filename generation
    output_file_name    = "Prasad_Bhagwat_UserBasedCF.txt"
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

    # Generating training data by subtracting testing data from it
    training_data   = input_data.subtractByKey(testing_data)

    # Creating RDD of tuples like ((user, movie), rating) and dictionary of (user, movie): rating
    usermovie_rating_RDD    = training_data.map(lambda x: ((x[0][0], x[0][1]), x[1]))
    usermovie_rating_dict   = dict(usermovie_rating_RDD.collect())#.toMap

    # Creating RDD of tuples like (user, movie) and dictionary of user: set(movies)
    user_movie_RDD          = training_data.map(lambda x: (x[0][0], x[0][1])).groupByKey().mapValues(set)
    user_movie_dict         = dict(user_movie_RDD.collect())#.toMap

    # Creating RDD of tuples like (user, movie) and dictionary of user: set(movies)
    movie_user_RDD          = training_data.map(lambda x: (x[0][1], x[0][0])).groupByKey().mapValues(set)
    movie_user_dict         = dict(movie_user_RDD.collect())# .toMap

    # Testing dictionary of user: movie
    testing_list            = testing.map(lambda x: (int(x.split(",")[0]), int(x.split(",")[1]))).collect()

    pearson_corelation      = [[0 for i in range(672)] for j in range(672)]
    output_list             = list()

    # Generating Pearson co-relation among all co-rated users for each movie
    for tuple in testing_list:
        user_1  = tuple[0]
        movie   = tuple[1]
        # Get all movies rated by User for which rating is to be predicted
        user1_movies            = user_movie_dict.get(user_1)
        user1_movies_list       = list()

        # Iterating over all the rated movies to generate list of the ratings given by User1
        for user1_movie in user1_movies:
            # Creating list of ratings of movies for User1
            user1_movies_list.append(usermovie_rating_dict[(user_1, user1_movie)])

        # Calculating average rating of User1 for all other movies
        user1_average_rating    = float(calculate_average(user1_movies_list))
        numerator               = 0.0
        denominator             = 0.0
        # Get all corated users for Movie for which rating is to be predicted
        corated_users           = movie_user_dict.get(movie)
        # Predict rating if co-rated user's list size more than 1
        if corated_users and len(corated_users) > 1:
            for user_2 in corated_users:
                intermediate_user2_movies       = user_movie_dict.get(user_2)
                user2_movies            	= copy.copy(intermediate_user2_movies)
                user2_movies_list       	= list()
                if movie in user2_movies:
                    user2_movies.remove(movie)
                # Iterating over all the rated movies to generate list of the ratings given by User1
                for user2_movie in user2_movies:
                    # Creating list of ratings of movies for User1
                    user2_movies_list.append(usermovie_rating_dict[(user_2, user2_movie)])
                # Calculating average rating of User1 for all other movies
                user2_average_rating    = float(calculate_average(user2_movies_list))

                corated_movies          = user1_movies.intersection(user2_movies)
                user1_corated_movies    = list()
                user2_corated_movies    = list()
                # Iterating over all the co-rated movies to generate list of the ratings given by User1 & User2
                for corated_movie in corated_movies:
                    # Creating list of ratings of movies for User1 & User2 on which the Pearson co-relation is to be calculated
                    user1_corated_movies.append(usermovie_rating_dict[(user_1, corated_movie)])
                    user2_corated_movies.append(usermovie_rating_dict[(user_2, corated_movie)])

                # Calculating Pearson co-relation of User1 & User2
                if len(user1_corated_movies) > 1:
                    pearson_corelation[user_1][user_2] = calculate_pearson_corelation(user1_corated_movies, user2_corated_movies)
                    # pearson_corelation[user_1][user_2] = cosine_similarity(user1_corated_movies, user2_corated_movies)
                else:
                    pearson_corelation[user_1][user_2] = 0

                if pearson_corelation[user_1][user_2] < 0.0:
                    continue
                else:
                    numerator   += float((usermovie_rating_dict[(user_2, movie)] - user2_average_rating) * pearson_corelation[user_1][user_2])
                    denominator += float(abs(pearson_corelation[user_1][user_2]))

            # Predicting rating using values of Pearson co-relation if denominator is not 0
            if denominator != 0:
                predicted_rating = user1_average_rating + float(numerator / denominator)
                # If predicted rating is within range (1, 5) then keep as it is
                if predicted_rating > 0.0 and predicted_rating <= 5.0:
                    output_list.append(((user_1, movie), predicted_rating))
                # If predicted rating is greater then 5 then truncate to 5
                elif predicted_rating > 5.0:
                    output_list.append(((user_1, movie), 5.0))
                # If predicted rating is lesser then 0 then truncate to 0
                elif predicted_rating <= 0.0:
                    output_list.append(((user_1, movie), 0.0))
            # Predicting rating as average of user rating if denominator is 0
            else:
                predicted_rating = user1_average_rating
                output_list.append(((user_1, movie), predicted_rating))
        # Predicting rating as average of user rating if there are no co-rated users available for the movie
        else:
            predicted_rating = user1_average_rating
            output_list.append(((user_1, movie), predicted_rating))

    # Generating output string as per expected format and writing in Output file
    output_result   = "\n".join(map(str, output_list)).replace("(", "").replace(")", "")
    output_file.write(output_result)
    output_file.close()

    # Joining actual ratings and predicted ratings for calculating RMSE
    output_rdd      = spark_context.parallelize(output_list)
    real_predicted  = input_data.map(lambda x: ((x[0][0], x[0][1]), x[1])).join(output_rdd)

    # Calculating and printing counts of error buckets
    differences     = real_predicted.map(lambda x: count_item_differences(x[1][0], x[1][1])).countByValue()
    for key in sorted(differences.iterkeys()):
        print "%s: %s" % (key, differences[key])

    # Calculating MSE and RMSE values and printing RMSE
    MSE             = float(real_predicted.map(lambda x: (x[1][0] - x[1][1]) ** 2).mean())
    RMSE            = float(math.sqrt(MSE))
    print "RMSE: ", RMSE

    # Printing time taken by the program
    print "Time:", int(time.time() - start_time), "sec"


# Entry point of the program
if __name__ == '__main__':
    main()
