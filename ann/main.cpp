//###begin<includes>
//noisy AutoencoderModel model and deep network
#include <shark/Models/FFNet.h>// neural network for supervised training
#include <shark/Models/Autoencoder.h>// the autoencoder to train unsupervised
#include <shark/Models/ImpulseNoiseModel.h>// model adding noise to the inputs
#include <shark/Models/ConcatenatedModel.h>// to concatenate Autoencoder with noise adding model

//training the  model
#include <shark/ObjectiveFunctions/ErrorFunction.h>//the error function performing the regularisation of the hidden neurons
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h> // squared loss used for unsupervised pre-training
#include <shark/ObjectiveFunctions/Loss/CrossEntropy.h> // loss used for supervised training
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h> // loss used for evaluation of performance
#include <shark/ObjectiveFunctions/Regularizer.h> //L1 and L2 regularisation
#include <shark/Algorithms/GradientDescent/SteepestDescent.h> //optimizer: simple gradient descent.
#include <shark/Algorithms/GradientDescent/Rprop.h> //optimizer for autoencoders

#include <shark/Data/Dataset.h>
#include <vector>
#include <shark/Data/Csv.h>
#include <shark/Algorithms/GradientDescent/CG.h>
#include <shark/Models/LinearModel.h>

#include <shark/Models/Normalizer.h>
#include <shark/Algorithms/Trainers/NormalizeComponentsUnitVariance.h>

#include<shark/Models/Softmax.h> //transforms model output into probabilities
#include<shark/Models/ConcatenatedModel.h> //provides operator >> for concatenating models

#include <shark/Core/Timer.h> //measures elapsed time

//###end<includes>

using namespace std;
using namespace shark;

LabeledData<RealVector, unsigned int> loadData(const std::string& dataFile,const std::string& labelFile){
    //we first load two separate data files for the training inputs and the labels of the data point
    Data<RealVector> inputs;
    Data<unsigned int> labels;
    try {
        importCSV(inputs, dataFile);
        importCSV(labels, labelFile);
    } catch (...) {
        cerr << "Unable to open file " <<  dataFile << " and/or " << labelFile << ". Check paths!" << endl;
        exit(EXIT_FAILURE);
    }
    //now we create a complete dataset which represents pairs of inputs and labels
    bool removeMean = true;
    Normalizer<RealVector> normalizer;
    NormalizeComponentsUnitVariance<RealVector> normalizingTrainer(removeMean);
    normalizingTrainer.train(normalizer, inputs);
    UnlabeledData<RealVector> normalizedData = transform(inputs, normalizer);
    LabeledData<RealVector, unsigned int> data(inputs, labels);
    exportCSV(inputs, "inputs.csv");
    return data;
}

void initialize_log(){
    ofstream inputFile;
    inputFile.open("log.csv", ios::out);
    if (inputFile.is_open()) {
        inputFile << "numInput" << "," << "numHiddenLayer" << "," << "numHidden" << "," << "unsupRegularisation"
        << "," << "noiseStrength" << "," << "unsupRegularisation" << "," << "regularisation" << "," << "iterations"
        << "," << "trainingerror" << "," << "testerror" << "," << "time" << endl;
    }
}

void add_log(const char* filename, size_t numHidden, size_t numHiddenLayer, double unsupRegularisation,
             double noiseStrength, size_t unsupIterations, double regularisation, size_t iterations,
             double trainingerror, double testerror, double time,
             size_t numInput){
    ofstream inputFile;
    inputFile.open(filename, std::ios_base::app);
    if (inputFile.is_open()) {
        cout << "Log file opened. " << endl;
        inputFile << numInput << "," << numHiddenLayer << "," << numHidden << "," << unsupRegularisation
        << "," << noiseStrength << "," << unsupRegularisation << "," << regularisation << "," << iterations
        << "," << trainingerror << "," << testerror << "," << time << endl;
    }
    else{
        cout << "Output log file does not exist. " << endl;
    }
}

//training of an auto encoder with one hidden layer
//###begin<function>
template<class AutoencoderModel>
AutoencoderModel trainAutoencoderModel(UnlabeledData<RealVector> const& data,//the data to train with
                                       std::size_t numHidden,//number of features in the AutoencoderModel
                                       double regularisation,//strength of the regularisation
                                       double noiseStrength, // strength of the added noise
                                       std::size_t iterations //number of iterations to optimize
){
    //###end<function>
    //###begin<model>
    //create the model
    std::size_t inputs = dataDimension(data);
    AutoencoderModel baseModel;
    baseModel.setStructure(inputs, numHidden);
    initRandomUniform(baseModel,-0.1*std::sqrt(1.0/inputs),0.1*std::sqrt(1.0/inputs));
    ImpulseNoiseModel noise(inputs,noiseStrength,0.0);//set an input pixel with probability p to 0
    ConcatenatedModel<RealVector,RealVector> model = noise>> baseModel;
    //###end<model>
    //###begin<objective>
    //create the objective function
    LabeledData<RealVector,RealVector> trainSet(data,data);//labels identical to inputs
    SquaredLoss<RealVector> loss;
    ErrorFunction error(trainSet, &model, &loss);
    TwoNormRegularizer regularizer(error.numberOfVariables());
    error.setRegularizer(regularisation,&regularizer);
    //###end<objective>
    //set up optimizer
    //###begin<optimizer>
    IRpropPlusFull optimizer;
    optimizer.init(error);
    cout << "Optimizing model: " + model.name() << endl;
    for(std::size_t i = 0; i != iterations; ++i){
        optimizer.step(error);
        cout << i << " " << optimizer.solution().value << endl;
    }
    //###end<optimizer>
    model.setParameterVector(optimizer.solution().point);
    return baseModel;
}

//###begin<network_types>
typedef Autoencoder<RectifierNeuron,LinearNeuron> AutoencoderModel;//type of autoencoder
typedef FFNet<RectifierNeuron,LinearNeuron> Network;//final supervised trained structure
//###end<network_types>

//unsupervised pre training of a network with two hidden layers
//###begin<pretraining_autoencoder>
Network unsupervisedPreTraining(UnlabeledData<RealVector> const& data,
                                std::size_t numHidden1,std::size_t numHidden2, std::size_t numOutputs,
                                double regularisation, double noiseStrength, std::size_t iterations){
    //train the first hidden layer
    cout << "Training first layer" << endl;
    AutoencoderModel layer =  trainAutoencoderModel<AutoencoderModel>(
                                                                      data,numHidden1,
                                                                      regularisation, noiseStrength,
                                                                      iterations
                                                                      );
    //compute the mapping onto the features of the first hidden layer
    UnlabeledData<RealVector> intermediateData = layer.evalLayer(0,data);
    
    //train the next layer
    cout << "Training second layer" << endl;
    AutoencoderModel layer2 =  trainAutoencoderModel<AutoencoderModel>(
                                                                       intermediateData,numHidden2,
                                                                       regularisation, noiseStrength,
                                                                       iterations
                                                                       );
    //###end<pretraining_autoencoder>
    //###begin<pretraining_creation>
    //create the final network
    Network network;
    network.setStructure(dataDimension(data),numHidden1,numHidden2, numOutputs);
    initRandomNormal(network,0.1);
    network.setLayer(0,layer.encoderMatrix(),layer.hiddenBias());
    network.setLayer(1,layer2.encoderMatrix(),layer2.hiddenBias());
    
    return network;
    //###end<pretraining_creation>
}

//unsupervised pre training of a network with multiple hidden layers
//###begin<pretraining_autoencoder>
Network unsupervisedPreTrainingMultipleLayers(UnlabeledData<RealVector> const& data,
                                              std::vector<size_t> const& layers, std::size_t numOutputs,
                                              double regularisation, double noiseStrength, std::size_t iterations){
    //train hidden layers
    vector<AutoencoderModel> layerVector;
    UnlabeledData<RealVector> intermediateData = data;
    for (size_t i = 1; i < layers.size()-1; ++i){
        cout << "Training layer " << i << endl;
        AutoencoderModel layer =  trainAutoencoderModel<AutoencoderModel>(intermediateData,layers[i],
                                                                          regularisation, noiseStrength,
                                                                          iterations);
        layerVector.push_back(layer);
        //compute the mapping onto the features of the first hidden layer
        if (i != layers.size() - 2) {
            intermediateData = layer.evalLayer(0,intermediateData);
        }
    }
    //###begin<pretraining_creation>
    //create the final network
    Network network;
    network.setStructure(layers);
    initRandomNormal(network,0.1);
    for(size_t j = 0; j < layers.size()-2; ++j){
        network.setLayer(j,layerVector[j].encoderMatrix(),layerVector[j].hiddenBias());
    }
    return network;
    //###end<pretraining_creation>
    //###end<pretraining_autoencoder>
}

void gridsearch(LabeledData<RealVector,unsigned int> const& data, LabeledData<RealVector,unsigned int> const& test){
    double minerror = 1;
    for (size_t i  = 0; i < 4; ++i){
        for (size_t j = 0; j <= 12; ++j){
            for (size_t k = 0; k < 3; ++k){
                std::size_t numHidden = 8 + 2*j; // number of hidden neurons for each hidden layer
                std::size_t numHiddenLayer = 2 + 2*i; // number of hidden layers
                
                //unsupervised hyper parameters
                double unsupRegularisation = 0.001; // default 0.001
                double noiseStrength = 0.055*k*k - 0.255*k + 0.3; // default 0.3, selection of 0.3, 0.1, and 0.01
                std::size_t unsupIterations = 100; // default 100
                
                //supervised hyper parameters
                double regularisation = 0.0001; // default 0.0001
                std::size_t iterations = 100; // default 100
                
                //set up hidden layer parameters
                vector<size_t> layer;
                layer.push_back(inputDimension(data));
                cout << "Inputs: " << inputDimension(data) << "\nOutputs: " << numberOfClasses(data) << endl;
                for (size_t i = 0; i < numHiddenLayer; ++i){
                    layer.push_back(numHidden);
                }
                layer.push_back(numberOfClasses(data));
                
                // unsupervised pre training for two hidden layer
                // Network network = unsupervisedPreTraining(data.inputs(),numHidden1, numHidden2,numberOfClasses(data), unsupRegularisation, noiseStrength, unsupIterations);
                
                // start timer
                Timer timer;
                
                //unsupervised pre training for multi layer deep neural network
                Network network = unsupervisedPreTrainingMultipleLayers(data.inputs(), layer, numberOfClasses(data),
                                                                        unsupRegularisation, noiseStrength, unsupIterations);
                
                //###begin<supervised_training>
                //create the supervised problem. Cross Entropy loss with one norm regularisation
                CrossEntropy loss;
                ErrorFunction error(data, &network, &loss);
                OneNormRegularizer regularizer(error.numberOfVariables());
                error.setRegularizer(regularisation,&regularizer);
                
                //optimize the model
                cout << "Training supervised model: " << endl;
                IRpropPlusFull optimizer;
                optimizer.init(error);
                for(size_t i = 0; i != iterations; ++i){
                    optimizer.step(error);
                    cout << i << " " << optimizer.solution().value << endl;
                }
                network.setParameterVector(optimizer.solution().point);
                //###end<supervised_training>
                
                //evaluation
                ZeroOneLoss<unsigned int,RealVector> loss01;
                Data<RealVector> predictionTrain = network(data.inputs());
                double trainingerror = loss01.eval(data.labels(), predictionTrain);
                cout << "Training set classification error: " << trainingerror << endl;
                
                Data<RealVector> prediction = network(test.inputs());
                double testerror = loss01.eval(test.labels(), prediction);
                cout << "Test set classification error: " << testerror << endl;
                
                // end timer
                double time  = timer.stop();
                string s3 = "Time cost: " + to_string(time);
                cout << s3 << endl;
                
                // output results
                if (testerror < minerror){
                    minerror = testerror;
                    exportCSV(data.inputs(), "inputs.csv");
                    exportCSV(predictionTrain, "predictionTrain.csv");
                    exportCSV(prediction, "prediction.csv");
                }
                
                //export log
                add_log("log.csv", numHidden, numHiddenLayer, unsupRegularisation, noiseStrength, unsupIterations,
                        regularisation, iterations, trainingerror, testerror, time, inputDimension(data));
            }
        }
    }
}

int main(){
    //load data
    string dataname = "datalag1.csv";
    string labelname = "lag1label.csv";
    LabeledData<RealVector,unsigned int> data = loadData(dataname, labelname);
    
    // shuffle data
    data.shuffle();
    
    // split into training set and test set
    LabeledData<RealVector,unsigned int> test =splitAtElement(data,static_cast<std::size_t>(0.5*data.numberOfElements()));
    
    //initialize log file
    initialize_log();
    
    //perform grid search to find the network that gives minimized error
    gridsearch(data, test);
    
    // END OF PROGRAM
    cout << "-- END OF PROGRAM --" << endl;
    return 666;
}
