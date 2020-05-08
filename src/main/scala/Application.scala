import classification.{DecisionTreeModel, LinearRegressionModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField, StructType}

object Application extends App {
	val spark = SparkSession.builder
	  .appName("Glass Classification With Naive Bayes")
	  .master("local[2]")
	  .getOrCreate()

	val schemaStruct = StructType(
		StructField("RI", DoubleType) ::
		  StructField("Na", DoubleType) ::
		  StructField("Mg", DoubleType) ::
		  StructField("Al", DoubleType) ::
		  StructField("Si", DoubleType) ::
		  StructField("K", DoubleType) ::
		  StructField("Ca", DoubleType) ::
		  StructField("Ba", DoubleType) ::
		  StructField("Fe", DoubleType) ::
		  StructField("Type", IntegerType) :: Nil
	)

	val df = spark.read
	  .schema(schemaStruct)
	  .option("header", value = true)
	  .csv("src/main/resources/glass.csv")
	  .na.drop()

	val Array(trainData, testData) = df.randomSplit(Array(0.9, 0.1))
	val regressionModel = LinearRegressionModel.getModel(schemaStruct, trainData, "Type")
	val regressionResult = regressionModel.transform(testData)

	val treeModel = DecisionTreeModel.getModel(schemaStruct, trainData, "Type")
	val treeResult = treeModel.transform(testData)

	val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy").setLabelCol("Type")

	println(s"Accuracy with Linear Regression = ${evaluator.evaluate(regressionResult)}")
	println(s"Accuracy with Decision Tree = ${evaluator.evaluate(treeResult)}")
}
