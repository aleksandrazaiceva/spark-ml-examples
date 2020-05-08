package classification

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.StructType

object LinearRegressionModel {
	def getModel(schema: StructType, trainData: DataFrame, labelColumn: String): PipelineModel = {
		val inputCols = schema.map(field => field.name).toArray
		val assembler = new VectorAssembler()
		  .setInputCols(inputCols)
		  .setOutputCol("features")

		val regression = new LinearRegression().setLabelCol("Type").setFeaturesCol("features").setMaxIter(50)
		val pipeline = new Pipeline()
		  .setStages(Array(assembler, regression))
		pipeline.fit(trainData)
	}
}
