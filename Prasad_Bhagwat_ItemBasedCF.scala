// Imports required for the program
import org.apache.spark.{SparkConf, SparkContext}
import java.io._
import scala.collection.mutable

object ItemBasedCF{

  // Counting error buckets
  def count_item_differences(actual: Float, predicted: Float): String= {
    val error = math.abs(actual - predicted)

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

  // Get nearest neighbors list in descending order based on jaccard similarities
  def get_nearest_neighbors(user_movies_list: mutable.Set[Int], movie_1: Int, jaccard_dict: Map[(Int, Int), Float]): List[Int]={
    val nearest_neighbors = new mutable.ListBuffer[(Int, Float)]
    for (movie <- user_movies_list){
      nearest_neighbors.append((movie, jaccard_dict.getOrElse((movie_1, movie), 0.toFloat)))
    }
    val neighbors = collection.immutable.ListMap(nearest_neighbors.sortWith(_._2 > _._2):_*)
    neighbors.keys.toList
  }


  // Main Function
  def main(args: Array[String]){
    val start_time = System.nanoTime()

    // Command Line Arguments
    val input_file      = args(0)
    val testing_file    = args(1)
    val jaccard_file    = args(2)

    // Output filename generation
    val output_file_name    = "Prasad_Bhagwat_ItemBasedCF.txt"
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

    // Reading jaccard similarity file
    val jaccard_data    = spark_context.textFile(jaccard_file)
    val jaccard_RDD     = jaccard_data.map( x => {
      val y = x.split(", ")
      ((y(0).toInt, y(1).toInt), y(2).toFloat)
    })
    val jaccard_dict    = jaccard_RDD.collect().toMap

    // Generating training data by subtracting testing data from it
    val training_data   = input_data.subtractByKey(testing_data)

    // Creating RDD of tuples like ((user, movie), rating) and dictionary of (user, movie): rating
    val usermovie_rating_RDD    = training_data.map(x => ((x._1._1, x._1._2), x._2))
    val usermovie_rating_dict   = usermovie_rating_RDD.collect().toMap

    // Creating RDD of tuples like (user, movie) and dictionary of user: set(movies)
    val user_movie_RDD          = training_data.map(x => (x._1._1, x._1._2)).groupByKey()//.mapValues(x => Set(x))
    val user_movie_dict         = user_movie_RDD.collect().toMap

    // Testing dictionary of user: movie
    val testing_list            = testing.map( x => {
      val y = x.split(",")
      (y(0).toInt, y(1).toInt)
    }).collect()

    var output_list             = new mutable.ListBuffer[((Int, Int), Float)]

    // Generating Pearson co-relation among all co-rated users for each movie
    for (tuple <- testing_list){
      val user      = tuple._1
      val movie_1   = tuple._2

      // Get all users who rated Movie for which rating is to be predicted
      val intermediate_user_movies= user_movie_dict.getOrElse(user, Iterable(0)).toSet
      val user_movies_set         = collection.mutable.Set(intermediate_user_movies.toArray:_*)
      val user_rated_movies_list  = new mutable.ListBuffer[Float]()
      // Iterating over all the rated movies to generate list of the ratings given by User1
      for (user_movie <- user_movies_set){
        // Creating list of ratings of movies for User1
        user_rated_movies_list.append(usermovie_rating_dict.getOrElse((user, user_movie), 0.toFloat))
      }
      val user_average     = calculate_average(user_rated_movies_list.toList)

      // Remove movie1 from movie1_users
      val user_movies_list = user_movies_set - movie_1
      val user_movies      = get_nearest_neighbors(user_movies_list, movie_1, jaccard_dict)

      // Calculating average rating of User1 for all other movies
      var numerator        = 0.0
      var denominator      = 0.0

      for (movie_2 <- user_movies.take(10)){
        if (movie_1 < movie_2){
          numerator   += (usermovie_rating_dict.getOrElse((user, movie_2), 0.toFloat) * jaccard_dict.getOrElse((movie_1, movie_2), 0.toFloat))
          denominator += math.abs(jaccard_dict.getOrElse((movie_1, movie_2), 0.toFloat))
        }
        else{
          numerator   += (usermovie_rating_dict.getOrElse((user, movie_2), 0.toFloat) * jaccard_dict.getOrElse((movie_2, movie_1), 0.toFloat))
          denominator += math.abs(jaccard_dict.getOrElse((movie_2, movie_1), 0.toFloat))
        }
      }

      // Predicting rating using values of Pearson co-relation if denominator is not 0
      if (denominator != 0.0){
        val predicted_rating = (numerator / denominator).toFloat
        // If predicted rating is within range (1, 5) then keep as it is
        if (predicted_rating > 0.0 && predicted_rating <= 5.0) {
          output_list.append(((user, movie_1), predicted_rating))
        }
        // If predicted rating is greater then 5 then truncate to 5
        else if (predicted_rating > 5.toFloat) {
          output_list.append(((user, movie_1), 5.toFloat))
        }
        // If predicted rating is lesser then 0 then truncate to 0
        else if (predicted_rating < 0.toFloat) {
          output_list.append(((user, movie_1), 0.toFloat))
        }
      }
      // Predicting rating as average of user rating if denominator is 0
      else{
        //val predicted_rating = user1_average_rating
        output_list.append(((user, movie_1), user_average))
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
