package classification

import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.StructType

object DecisionTreeModel {
	def getModel(schema: StructType, trainData: DataFrame, labelColumn: String): PipelineModel = {
		val inputCols = schema.map(field => field.name).toArray
		val assembler = new VectorAssembler()
		  .setInputCols(inputCols)
		  .setOutputCol("features")

		val tree = new DecisionTreeClassifier()
		  .setLabelCol(labelColumn)
		  .setFeaturesCol("features")

		val pipeline = new Pipeline()
		  .setStages(Array(assembler, tree))
		pipeline.fit(trainData)
	}
}
