# Imports required for the program
from pyspark import SparkConf, SparkContext
import time
import sys
import math
from pyspark.mllib.recommendation import ALS, Rating


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


# Main Function
def main():
    start_time = time.time()

    # Command Line Arguments
    input_file      = sys.argv[1]
    testing_file    = sys.argv[2]

    # Output filename generation
    output_file_name    = "Prasad_Bhagwat_ModelBasedCF.txt"
    output_file         = open(output_file_name, "w")

    # Creating Spark Context
    spark_config    = SparkConf()
    spark_context   = SparkContext(conf=spark_config)
    spark_context.setLogLevel("WARN")

    # Reading input training data file and extracting header
    input           = spark_context.textFile(input_file).filter(lambda x: "userId" not in x)
    input_data      = input.map(lambda x: ((int(x.split(",")[0]), int(x.split(",")[1])), float(x.split(",")[2])))

    # Reading test data file and extracting header
    testing         = spark_context.textFile(testing_file).filter(lambda x: "userId" not in x)
    testing_data    = testing.map(lambda x: ((int(x.split(",")[0]), int(x.split(",")[1])), None))

    # Generating training data by subtracting testing data from it
    training_data   = input_data.subtractByKey(testing_data)

    # Configuration parameters for training model
    user_rating     = training_data.map(lambda x: Rating(int(x[0][0]), int(x[0][1]), float(x[1])))
    rank            = 12    # Can be changed for better performance
    num_iterations  = 12    # Can be changed for better performance
    lambda_value    = 0.1   # Can be changed for better performance
    num_blocks      = 1     # Can be changed for better performance
    seed            = 4     # Can be changed for better performance

    # ALS Model training
    ALS_model       = ALS.train(ratings= user_rating, \
                                rank= rank, \
                                iterations= num_iterations, \
                                lambda_= lambda_value, \
                                blocks= num_blocks, \
                                seed= seed)

    # Generating testing data
    user_movies     = testing_data.map(lambda x: (int(x[0][0]), int(x[0][1])))

    # Predicted values for testing data
    predicted_vals  = ALS_model.predictAll(user_movies).map(lambda x: ((x[0], x[1]), x[2]))

    # Generating output string as per expected format and writing in Output file
    output_list     = predicted_vals.sortByKey().map(lambda x: (x[0][0], x[0][1], x[1]))
    output_result   = "\n".join(map(str, output_list.collect())).replace("(", "").replace(")", "")
    output_file.write(output_result)
    output_file.close()

    # Joining actual ratings and predicted ratings for calculating RMSE
    real_predicted  = input_data.map(lambda x: ((x[0][0], x[0][1]), x[1])).join(predicted_vals)

    # Calculating and printing counts of error buckets
    differences     = real_predicted.map(lambda x : count_item_differences(x[1][0], x[1][1])).countByValue()
    for key in sorted(differences.iterkeys()):
        print "%s: %s" % (key, differences[key])

    # Calculating MSE and RMSE values and printing RMSE
    MSE             = float(real_predicted.map(lambda x: (x[1][0] - x[1][1]) ** 2).mean())
    RMSE            = float(math.sqrt(MSE))
    print "RMSE:", RMSE

    # Printing time taken by the program
    print "Time:", int(time.time() - start_time), "sec"


# Entry point of the program
if __name__ == '__main__':
    main()
