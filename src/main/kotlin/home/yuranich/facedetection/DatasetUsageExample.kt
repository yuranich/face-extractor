package home.yuranich.facedetection

import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.Logger
import org.slf4j.LoggerFactory


fun main(args: Array<String>) {
    //number of rows and columns in the input pictures
    val numRows = 28
    val numColumns = 28
    val outputNum = 10 // number of output classes
    val batchSize = 128 // batch size for each epoch
    val rngSeed = 123 // random number seed for reproducibility
    val numEpochs = 15 // number of epochs to perform

    val conf: MultiLayerConfiguration = NeuralNetConfiguration.Builder()
        .seed(rngSeed) //include a random seed for reproducibility
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) // use stochastic gradient descent as an optimization algorithm
        .iterations(1) // in nearly all cases should be 1
        .learningRate(0.006) //specify the learning rate
        .updater(Updater.NESTEROVS).momentum(0.9) //specify the rate of change of the learning rate.
        .regularization(true).l2(1e-4)
        .list()
        .layer(0, DenseLayer.Builder() //create the first, input layer with xavier initialization
            .nIn(numRows * numColumns)
            .nOut(1000)
            .activation(Activation.RELU)
            .weightInit(WeightInit.XAVIER)
            .build())
        .layer(1, OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
            .nIn(1000)
            .nOut(outputNum)
            .activation(Activation.SOFTMAX)
            .weightInit(WeightInit.XAVIER)
            .build())
        .build()

    val model = MultiLayerNetwork(conf)
    model.init()
    //print the score with every 1 iteration
    model.setListeners(ScoreIterationListener(1))

    //Get the DataSetIterators:
    val mnistTrain = MnistDataSetIterator(batchSize, true, rngSeed)
    val mnistTest = MnistDataSetIterator(batchSize, false, rngSeed)

    // the "simple" way to do multiple epochs is to wrap fit() in a loop
    (1 .. numEpochs).forEach {  model.fit(mnistTrain) }

    val evaluation = model.evaluate(mnistTest)

// print the basic statistics about the trained classifier
    println("Accuracy: "+evaluation.accuracy())
    println("Precision: "+evaluation.precision())
    println("Recall: "+evaluation.recall())

// in more complex scenarios, a confusion matrix is quite helpful
    println(evaluation.confusionToString())
}