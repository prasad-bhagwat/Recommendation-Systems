// Imports required for the program
import org.apache.spark.{SparkConf, SparkContext}
import java.io._
import org.apache.spark.mllib.recommendation.{ALS, Rating}


object ModelBasedCF{

  // Counting error buckets
  def count_item_differences(actual: Float, predicted: Float): String= {
    val error = (math.abs(actual - predicted)).toFloat

    if (error >= 0.toFloat && error < 1.toFloat) {
      ">= 0 and <1"
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

  // Main Function
  def main(args: Array[String]){
    val start_time = System.nanoTime()

    // Command Line Arguments
    val input_file      = args(0)
    val testing_file    = args(1)

    // Output filename generation
    val output_file_name    = "Prasad_Bhagwat_ModelBasedCF.txt"
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

    //println(training_data)
    // Configuration parameters for training model
    val user_rating     = training_data.map(x => Rating(x._1._1, x._1._2, x._2))
    val rank            = 12
    val number_iter     = 12
    val lambda_val      = 0.1
    val blocks          = 1
    val seed            = 4

    // ALS Model training
    val ALS_model       = ALS.train(ratings = user_rating,
                          rank= rank,
                          iterations= number_iter,
                          lambda= lambda_val,
                          blocks= blocks,
                          seed= seed)

    // Generating testing data
    val user_movies     = testing_data.map(x=> (x._1._1, x._1._2))

    // Predicted values for testing data
    val predicted_vals  = ALS_model.predict(user_movies).map{case Rating(user, product, rating) => ((user, product), rating)}

    // Generating output string as per expected format and writing in Output file
    val output_list     = predicted_vals.sortByKey().map(x=> (x._1._1, x._1._2, x._2)).collect()

    // Generating expected output of the form "User, Movie, Predicted Rating"
    val temp_result     = output_list.mkString("\n")
    val final_result    = temp_result.replace("(", "").replace(")", "").replace(",", ", ")

    // Writing results into output file
    output_file.write(final_result)
    output_file.close()

    // Joining actual ratings and predicted ratings for RMSE calculations
    val real_predicted  = input_data.map(x=> ((x._1._1, x._1._2), x._2)).join(predicted_vals)

    // Calculating and printing counts of error buckets
    val differences     = real_predicted.map(x=> count_item_differences(x._2._1, x._2._2.toFloat)).countByValue()
    for ((key, value) <- differences.toArray.sorted){
      println(key + ": " + value)
    }

    // Calculating MSE and RMSE values and printing RMSE
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