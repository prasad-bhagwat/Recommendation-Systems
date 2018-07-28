// Imports required for the program
import org.apache.spark.{SparkConf, SparkContext}
import java.io._
import util.control.Breaks._
import scala.collection.mutable.{ListBuffer, ArrayBuffer}


object UserBasedCF{

  // Counting error buckets
  def count_item_differences(actual: Float, predicted: Float): String= {
    val error = (math.abs(actual - predicted))

    if (error >= 0.toFloat && error < 1.toFloat) {
      ">=0 and <1"
    }
    else if (error >= 1.toFloat && error < 2.toFloat) {
      ">=1 and <2"
    }
    else if (error >= 2.toFloat && error < 3.toFloat) {
      ">=2 and <3"
    }
    else if (error >= 3.toFloat && error < 4.toFloat) {
      ">=3 and <4"
    }
    else {
      ">=4"
    }
  }

  // Calculating average of input list
  def calculate_average(input_list: List[Float]): Float= {
    if (input_list.nonEmpty){
      input_list.sum / input_list.length
    }
    else{
      0.toFloat
    }
  }


  // Pearson co-relation calculation
  def calculate_pearson_corelation(list_1: List[Float], list_2: List[Float]): Float={
    // Calculating average of both lists
    val list_1_avg      = calculate_average(list_1)
    val list_2_avg      = calculate_average(list_2)
    var numerator       = 0.toFloat
    var list_1_diff_sum = 0.toFloat
    var list_2_diff_sum = 0.toFloat
    // Calculating numerator for Pearson co-relation
    for (item <- list_1.indices){
      val list_1_diff     = list_1(item) - list_1_avg
      val list_2_diff     = list_2(item) - list_2_avg
      numerator           += list_1_diff * list_2_diff
      list_1_diff_sum     += list_1_diff * list_1_diff
      list_2_diff_sum     += list_2_diff * list_2_diff
    }
    // Calculating denominator for Pearson co-relation
    val denominator   = list_1_diff_sum * list_2_diff_sum

    // Returning Pearson Co-relation
    if (denominator != 0.toFloat){
      (numerator / math.sqrt(denominator)).toFloat
    }
    else{
      0.toFloat
    }
  }


  // Main Function
  def main(args: Array[String]){
    val start_time = System.nanoTime()

    // Command Line Arguments
    val input_file      = args(0)
    val testing_file    = args(1)

    // Output filename generation
    val output_file_name    = "Prasad_Bhagwat_UserBasedCF.txt"
    val output_file         = new PrintWriter(new File(output_file_name))

    // Creating Spark Context
    val spark_config    = new SparkConf()
    val spark_context   = new SparkContext(spark_config)
    spark_context.setLogLevel("WARN")

    // Reading input training data file and extracting header
    val input           = spark_context.textFile(input_file).filter(x => ! x.contains("userId"))
    val input_data      = input.map( x => {
      val y = x.split(",")
      ((y(0).toInt, y(1).toInt), y(2).toFloat)
    })

    // Reading test data file and extracting header
    val testing         = spark_context.textFile(testing_file).filter(x => ! x.contains("userId"))
    val testing_data    = testing.map( x => {
      val y = x.split(",")
      ((y(0).toInt, y(1).toInt), None)
    })

    // Generating training data by subtracting testing data from it
    val training_data   = input_data.subtractByKey(testing_data)

    // Creating RDD of tuples like ((user, movie), rating) and dictionary of (user, movie): rating
    val usermovie_rating_RDD    = training_data.map(x => ((x._1._1, x._1._2), x._2))
    val usermovie_rating_dict   = usermovie_rating_RDD.collect().toMap

    // Creating RDD of tuples like (user, movie) and dictionary of user: set(movies)
    val user_movie_RDD          = training_data.map(x => (x._1._1, x._1._2)).groupByKey()//.mapValues(x => Set(x))
    val user_movie_dict         = user_movie_RDD.collect().toMap

    // Creating RDD of tuples like (user, movie) and dictionary of user: set(movies)
    val movie_user_RDD          = training_data.map(x => (x._1._2, x._1._1)).groupByKey()//.mapValues(x => Set(x))
    val movie_user_dict         = movie_user_RDD.collect().toMap

    // Testing dictionary of user: movie
    val testing_list            = testing.map( x => {
      val y = x.split(",")
      ((y(0).toInt, y(1).toInt))
    }).collect()

    var pearson_corelation      = ArrayBuffer.fill(672, 672)(0.toFloat)
    var output_list             = new ListBuffer[((Int, Int), Float)]

    // Generating Pearson co-relation among all co-rated users for each movie
    for (tuple <- testing_list){
      val user_1  = tuple._1
      val movie   = tuple._2

      // Get all movies rated by User for which rating is to be predicted
      val user1_movies            = user_movie_dict.getOrElse(user_1, Iterable(0)).toSet
      val user1_movies_list       = new ListBuffer[Float]()

      // Iterating over all the rated movies to generate list of the ratings given by User1
      for (user1_movie <- user1_movies){
          // Creating list of ratings of movies for User1
          user1_movies_list += usermovie_rating_dict.getOrElse((user_1, user1_movie), 0)
      }

      // Calculating average rating of User1 for all other movies
      val user1_average_rating    = calculate_average(user1_movies_list.toList)
      var numerator               = 0.0
      var denominator             = 0.0

      // Get all corated users for Movie for which rating is to be predicted
      val corated_users           = movie_user_dict.getOrElse(movie, Iterable(0)).toSet
      // Predict rating if co-rated user's list size more than 1
      if (corated_users.size > 1){
        for (user_2 <- corated_users){
          val intermediate_user2_movies = user_movie_dict.getOrElse(user_2, Iterable(0)).toSet
          val user_2_movies       	= collection.mutable.Set(intermediate_user2_movies.toArray:_*)
          val user2_movies        	= user_2_movies.map(x => x)
          val user2_movies_list   	= new ListBuffer[Float]()
          if (user2_movies.contains(movie)){
            user2_movies -= movie
          }
          // Iterating over all the rated movies to generate list of the ratings given by User1
          for (user2_movie <- user2_movies){
            // Creating list of ratings of movies for User1
            user2_movies_list.append(usermovie_rating_dict.getOrElse((user_2, user2_movie), 0.toFloat))
          }
          // Calculating average rating of User1 for all other movies
          val user2_average_rating    = calculate_average(user2_movies_list.toList)

          val corated_movies          = user1_movies.intersect(user2_movies)
          val user1_corated_movies    = new ListBuffer[Float]()
          val user2_corated_movies    = new ListBuffer[Float]()
          // Iterating over all the co-rated movies to generate list of the ratings given by User1 & User2
          for (corated_movie <- corated_movies){
            // Creating list of ratings of movies for User1 & User2 on which the Pearson co-relation is to be calculated
            user1_corated_movies.append(usermovie_rating_dict.getOrElse((user_1, corated_movie), 0.toFloat))
            user2_corated_movies.append(usermovie_rating_dict.getOrElse((user_2, corated_movie), 0.toFloat))
          }
          // Calculating Pearson co-relation of User1 & User2
          if (user1_corated_movies.size > 1) {
            pearson_corelation(user_1)(user_2) = calculate_pearson_corelation(user1_corated_movies.toList, user2_corated_movies.toList)
            // pearson_corelation[user_1][user_2] = cosine_similarity(user1_corated_movies, user2_corated_movies)
          }
          else{
            pearson_corelation(user_1)(user_2) = 0.toFloat
          }

          breakable {
            if (pearson_corelation(user_1)(user_2) < 0.toFloat) {
              // break out of the 'breakable', continue the outside loop
              break
            }
            else {
              numerator   += ((usermovie_rating_dict.getOrElse((user_2, movie), 0.toFloat) - user2_average_rating) * pearson_corelation(user_1)(user_2)).toFloat
              denominator += Math.abs(pearson_corelation(user_1)(user_2))
            }
          }
        }
        // Predicting rating using values of Pearson co-relation if denominator is not 0
        if (denominator != 0){
          val predicted_rating = user1_average_rating + (numerator / denominator).toFloat
          // If predicted rating is within range (1, 5) then keep as it is
          if (predicted_rating > 0.0 && predicted_rating <= 5.0) {
            output_list.append(((user_1, movie), predicted_rating))
          }
          // If predicted rating is greater then 5 then truncate to 5
          else if (predicted_rating > 5.toFloat) {
            output_list.append(((user_1, movie), 5.toFloat))
          }
          // If predicted rating is lesser then 0 then truncate to 0
          else if (predicted_rating < 0.toFloat) {
            output_list.append(((user_1, movie), 0.toFloat))
          }
        }
        // Predicting rating as average of user rating if denominator is 0
        else{
          //val predicted_rating = user1_average_rating
          output_list.append(((user_1, movie), user1_average_rating))
        }
      }
      // Predicting rating as average of user rating if there are no co-rated users available for the movie
      else{
        output_list.append(((user_1, movie), user1_average_rating))
      }
    }

    // Generating expected output of the form "User, Movie, Predicted Rating"
    val intermediate_result     = output_list.mkString("\n")
    val final_result    	= intermediate_result.replace("(", "").replace(")", "").replace(",", ", ")

    // Writing results into output file
    output_file.write(final_result)
    output_file.close()

    // Joining actual ratings and predicted ratings for RMSE calculations
    val output_rdd      = spark_context.parallelize(output_list.toList)
    val real_predicted  = input_data.map(x => ((x._1._1, x._1._2), x._2)).join(output_rdd)

    // Calculating and printing counts of error buckets
    val differences     = real_predicted.map(x=> count_item_differences(x._2._1, x._2._2.toFloat)).countByValue()
    for ((key, value) <- differences.toArray.sorted){
      println(key + ": " + value)
    }

    // Calculating MSE and RMSE values
    val MSE             = real_predicted.map{ x =>
      val err = x._2._1 - x._2._2.toFloat
      err * err
    }.mean()

    val RMSE            = math.sqrt(MSE)
    println("RMSE: " + RMSE)

    // Printing time taken by program
    println("Time: "+ ((System.nanoTime() - start_time) / 1e9d).toInt + " sec")
  }
}
