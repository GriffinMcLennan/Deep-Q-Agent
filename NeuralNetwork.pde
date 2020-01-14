import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.util.ArrayList;
import java.util.List;
import java.io.File;
import java.io.FileWriter;
import java.io.FileReader;

public class NeuralNetwork
{
    //parameter
    double lr;

    double[] xIn;
    double[] yIn;

    Matrix input;
    Matrix y;
    Matrix m;

    //general info
    List<Matrix> weights;
    List<Matrix> bias;
    List<Matrix> z;
    List<Matrix> layers;
    List<String> activations;

    //updates
    List<Matrix> d_weights;
    List<Matrix> d_bias;

    boolean inputLayerAdded;

    int epoch;
    int print_every_x_epoch;

    /**
     * Constructor to initialize the Neural Network
     * @param learning_rate     Learning rate for the gradient-descent
     * @param print_every_x_epoch   Value to specify how often to print the value of the cost function.
     */
    public NeuralNetwork(double learning_rate, int print_every_x_epoch)
    {
        weights = new ArrayList<Matrix>();
        bias = new ArrayList<Matrix>();
        z = new ArrayList<Matrix>();
        layers = new ArrayList<Matrix>();
        activations = new ArrayList<String>();

        //updates
        d_weights = new ArrayList<Matrix>();
        d_bias = new ArrayList<Matrix>();

        m = new Matrix();
        epoch = 1;
        this.inputLayerAdded = false;
        this.lr = learning_rate;
        this.print_every_x_epoch = print_every_x_epoch;
    }


    /**
     * Specifies the input layer dimensions for the NN.
     * @param inputShape    Dimension of input vector
     * @param outputShape   Dimension of output vector
     * @param activation_function   activation function for the layer
     */
    public void addInputLayer(int inputShape, int outputShape, String activation_function)
    {
        if (inputLayerAdded)
        {
            System.out.println("ERROR: Input layer already added!");
            System.exit(-1);
        }
        else if (!m.validFunction(activation_function))
        {
            System.out.println("ERROR: Invalid activation function for input layer!");
            System.exit(-1);
        }
        
        inputLayerAdded = true;

        //add weights, biases, and activation function
        weights.add((new Matrix(inputShape, outputShape)));
        bias.add(new Matrix(1, outputShape));
        activations.add(activation_function);

        //Add matrices to maintain matching array dimensions
        z.add(new Matrix());
        layers.add(new Matrix());
        d_weights.add(new Matrix());
        d_bias.add(new Matrix());
    }

    /**
     * Method to add a non-input layer
     * @param outputShape   Dimension of output vector
     * @param activation_function   activation function for the layer
     */
    public void addLayer(int outputShape, String activation_function)
    {
        if (!inputLayerAdded)
        {
            System.out.println("ERROR: Input layer hasn't been added!");
            System.exit(-1);
        }
        else if (!m.validFunction(activation_function))
        {
            System.out.println("ERROR: Invalid activation function for input layer!");
            System.exit(-1);
        }

        //Find the shape of the previous layer
        int lastLayerIndex = weights.size() - 1; 
        double[][] currentLayerWeights = weights.get(lastLayerIndex).getWeights();
        int currentShape = currentLayerWeights[0].length;
        
        //add weights, biases, and activation function
        weights.add(new Matrix(currentShape, outputShape));
        bias.add(new Matrix(1, outputShape));
        activations.add(activation_function);

        //Add matrices to maintain matching array dimensions
        z.add(new Matrix());
        layers.add(new Matrix());
        d_weights.add(new Matrix());
        d_bias.add(new Matrix());
    }

    /**
     * Forward propagate an input through the Neural Network.
     * @param inputArray    Array of inputs to be fed to the first set of weights
     */
    public void FeedForward(double[] inputArray, boolean printOutput)
    {
        Matrix input = new Matrix(inputArray);

        //Calculating the first layer's output using the input vector
        z.set(0, m.ElementWiseAddition(m.MatrixMultiply(input, weights.get(0)), bias.get(0)));

        layers.set(0, m.stringToFunctionMap(activations.get(0), z.get(0), false));

        //loop over the remaining layers
        for (int j = 1; j < weights.size(); j++)
        {
            z.set(j, m.ElementWiseAddition(m.MatrixMultiply(layers.get(j - 1), weights.get(j)), bias.get(j)));

            layers.set(j, m.stringToFunctionMap(activations.get(j), z.get(j), false));
        }

        //Optional: print results
        if (printOutput)
        {
            System.out.println(layers.get(layers.size() - 1));
        }
    }

    /**
     * Method to train the Neural Network.
     * @param inputArray        Two-dimensional array of input data
     * @param outputArray       Two-dimensional array of output data
     * @param num_epochs        Number of epochs to train
     * @param algorithm         Algorithm to use
     */
    public void train(double[][] inputArray, double[][] outputArray, int num_epochs, String algorithm)
    {
        for (int i = 1; i <= num_epochs; i++)
        {
            if (algorithm.equals("gd"))
            {
                GD(inputArray, outputArray);
            }
            else if (algorithm.equals("sgd"))
            {
                SGD(inputArray, outputArray);
            }
        }
    }

    /**
     * Method to perform backpropagation to calculate the gradients for every weight in the network.
     * 
     * @param inputArray    Array of input data to be fed through the network
     * @param outputArray   Array of output data.
     */
    public void BackPropagate(double[] inputArray, double[] outputArray)
    {
        FeedForward(inputArray, false);

        input = new Matrix(inputArray);

        y = new Matrix(outputArray);
        
        //cache Matrix to perform memoization. Keeps track of all the previous partial derivatives.
        Matrix cache; 

        //Calculate change in weights/bias for output layer
        int outLayerInd = layers.size() - 1;
        cache = m.ElementWiseMultiply(m.ElementWiseSubtraction(y, layers.get(outLayerInd)), m.stringToFunctionMap(activations.get(outLayerInd), z.get(outLayerInd), true));

        d_weights.set(outLayerInd, m.MatrixMultiply(layers.get(outLayerInd - 1).transpose(), cache));
        d_bias.set(outLayerInd, cache);

        

        //calculate change in weights/bias for layers [2, outputLayer)
        for (int j = layers.size() - 1; j > 1; j--)
        {
            cache = m.ElementWiseMultiply(m.MatrixMultiply(cache, weights.get(j).transpose()), m.stringToFunctionMap(activations.get(j - 1), z.get(j - 1), true));
            d_weights.set(j - 1, m.MatrixMultiply(layers.get(j - 2).transpose(), cache));
            d_bias.set(j - 1, cache);
        }


        //caclculate change in weights/bias for the first layer
        cache = m.ElementWiseMultiply(m.MatrixMultiply(cache, weights.get(1).transpose()), m.stringToFunctionMap(activations.get(0), z.get(0), true));
        d_weights.set(0, m.MatrixMultiply(input.transpose(), cache));
        d_bias.set(0, cache);
    }

    /**
     * Perform stochastic gradient descent on a set of inputs with their corresponding 
     * output values.
     * @param inputArray    Array of input values   
     * @param outputArray   Array of output values for the Neural Net to attempt to predict
     */
    public void SGD(double[][] inputArray, double[][] outputArray)
    {

        if (inputArray.length != outputArray.length)
        {
            System.out.println("ERROR: Number of inputs doesn't match number of outputs for backpropagation!");
            System.exit(-1);
        }
        
        double cost = 0;

        //loop over each element in the inputArray
        for (int i = 0; i < inputArray.length; i++)
        {
            BackPropagate(inputArray[i], outputArray[i]);

            if (epoch % print_every_x_epoch == 0)
            {
                Matrix MSE = m.ElementWiseSubtraction(y, layers.get(layers.size() - 1));
                MSE = m.ElementWiseMultiply(MSE, MSE);
                cost += MSE.sumMatrix();
            }

            //adjust the weights and biases
            for (int j = 0; j < weights.size(); j++)
            {
                d_weights.set(j, m.scaleMatrix(d_weights.get(j), lr));
                d_bias.set(j, m.scaleMatrix(d_bias.get(j), lr));

                weights.set(j, m.ElementWiseAddition(weights.get(j), d_weights.get(j)));
                bias.set(j, m.ElementWiseAddition(bias.get(j), d_bias.get(j)));
            }

            //Optional: Print the output of the neural network
            //System.out.println(layers.get(layers.size() - 1));
        }

        if (epoch % print_every_x_epoch == 0)
        {
            cost = cost / inputArray.length;
            System.out.printf("Epoch %d, cost = %.14f\n", epoch, cost);
        }
        epoch++;
    }

    /**
     * Perform gradient descent on a set of inputs with their corresponding outputs to minimize the cost function
     * @param inputArray    Two-dimensional array of input data
     * @param outputArray   Two-dimensional array of output data
     */
    public void GD(double[][] inputArray, double[][] outputArray)
    {
        List<Matrix> d_weights_avg = new ArrayList<Matrix>();
        List<Matrix> d_bias_avg = new ArrayList<Matrix>();

        if (inputArray.length != outputArray.length)
        {
            System.out.println("ERROR: Number of inputs doesn't match number of outputs for backpropagation!");
            System.exit(-1);
        }

        for (int i = 0; i < d_weights.size(); i++)
        {
            double[][] weightSize = weights.get(i).getWeights();
            int m = weightSize.length;
            int n = weightSize[0].length;

            //initialize to all zeros
            d_weights_avg.add(new Matrix(new double[m][n]));
            d_bias_avg.add(new Matrix(new double[1][n]));
        }

        double cost = 0;

        //loop over each element in the inputArray
        for (int i = 0; i < inputArray.length; i++)
        {
            BackPropagate(inputArray[i], outputArray[i]);

            if (epoch % print_every_x_epoch == 0)
            {
                Matrix MSE = m.ElementWiseSubtraction(y, layers.get(layers.size() - 1));
                MSE = m.ElementWiseMultiply(MSE, MSE);
                cost += MSE.sumMatrix();
            }

            //add change in weights and biases to total difference
            for (int j = 0; j < weights.size(); j++)
            {
                d_weights_avg.set(j, m.ElementWiseAddition(d_weights_avg.get(j), d_weights.get(j)));
                d_bias_avg.set(j, m.ElementWiseAddition(d_bias_avg.get(j), d_bias.get(j)));

            }

            //Optional: Print the output of the neural network
            //System.out.println(layers.get(layers.size() - 1));
        }

        double numSamplesInverse = 1 / (double)inputArray.length;

        //average gradients:
        for (int i = 0; i < d_weights_avg.size(); i++)
        {
            d_weights_avg.set(i, m.scaleMatrix(d_weights_avg.get(i), numSamplesInverse * lr));
            d_bias_avg.set(i, m.scaleMatrix(d_bias_avg.get(i), numSamplesInverse * lr));

            weights.set(i, m.ElementWiseAddition(weights.get(i), d_weights_avg.get(i)));
            bias.set(i, m.ElementWiseAddition(bias.get(i), d_bias_avg.get(i)));
        }

        if (epoch % print_every_x_epoch == 0)
        {
            cost = cost / inputArray.length;
            System.out.printf("Epoch %d, cost = %.14f\n", epoch, cost);
        }
        epoch++;
    }

    /**
     * Method to save the current Neural Network to a text file to be reloaded later on.
     * @param networkName   The name of the Neural Network that will be used to identify the 
     *                      text file.
     */
    public void saveNetwork(String networkName)
    {
        System.out.printf("Attempting to save Neural Network: %s\n", networkName);

        try
        {
            //Create the Buffered Writer
            File file = new File(networkName + ".txt");
            FileWriter fr = new FileWriter(file);
            BufferedWriter br = new BufferedWriter(fr);

            //write info about how many layers the network has.
            br.write("" + weights.size());
            br.newLine();

            //loop over all of the weights and write their values to the text file.
            for (int i = 0; i < weights.size(); i++)
            {
                double[][] layer_weights = weights.get(i).getWeights();

                br.write("" + layer_weights.length + " " + layer_weights[0].length + " " + activations.get(i));
                br.newLine();

                for (int j = 0; j < layer_weights.length; j++)
                {
                    for (int k = 0; k < layer_weights[0].length; k++)
                    {
                        br.write(layer_weights[j][k] + " ");
                    }

                    br.newLine();
                }

                //get the weights of the current bias layer now
                double[][] bias_weights = bias.get(i).getWeights();

                //write them to the text file.
                for (int j = 0; j < bias_weights[0].length; j++)
                {
                    br.write(bias_weights[0][j] +" ");
                }

                br.newLine();

            }

            br.close();
            fr.close();

            System.out.printf("Successfully saved Neural Network: %s\n", networkName);
        }
        catch(Exception e)
        {
            System.out.println("ERROR: Saving Network!");
            System.out.println(e);
        }
    }

    /**
     * Method to load a previously saved Neural Network.
     * @param networkName   The name of the previous neural network to load.
     */
    public void loadNetwork(String networkName)
    {
        networkName = networkName + ".txt";

        //reset all values.
        weights = new ArrayList<Matrix>();
        bias = new ArrayList<Matrix>();
        z = new ArrayList<Matrix>();
        layers = new ArrayList<Matrix>();
        activations = new ArrayList<String>();

        //set inputLayerAdded to true. Can't load a network without an input layer.
        inputLayerAdded = true;

        d_weights = new ArrayList<Matrix>();
        d_bias = new ArrayList<Matrix>();


        try
        {
            System.out.printf("Attempting to load Neural Network: %s\n", networkName.substring(0, networkName.length() - 4));

            File file = new File(networkName);
            FileReader fr = new FileReader(file);
            BufferedReader br = new BufferedReader(fr);

            //get the number of weight layers.
            int num_weight_layers = Integer.parseInt(br.readLine());

            //loop over each layer
            for (int i = 0; i < num_weight_layers; i++)
            {
               String weight_dimensions = br.readLine().trim();
               String[] rowsColsActivation = weight_dimensions.split(" "); 
               
               int rows = Integer.parseInt(rowsColsActivation[0]);
               int cols = Integer.parseInt(rowsColsActivation[1]);
               String activation = rowsColsActivation[2];

               String[][] row_weights = new String[rows][cols];

               //loop over every row and column writing the value to the row_weights array.
               for (int j = 0; j < rows; j++)
               {
                    String current_row_weights = br.readLine();
                    String[] vector_current_row_weights = current_row_weights.split(" ");

                    for (int k = 0; k < cols; k++)
                    {
                        row_weights[j][k] = vector_current_row_weights[k];
                    }
                }

                //load bias now
                String current_bias_weights = br.readLine();
                String[] vector_current_bias_weights = current_bias_weights.split(" ");

                //use Matrix class constructors to create matrices from the string arrays.
                Matrix currentLayerWeights = new Matrix(row_weights);
                Matrix currentBiasWeights = new Matrix(vector_current_bias_weights);

                //add Matrices and activation function.
                weights.add(currentLayerWeights);
                bias.add(currentBiasWeights);
                activations.add(activation);

                z.add(new Matrix());
                layers.add(new Matrix());
                d_weights.add(new Matrix());
                d_bias.add(new Matrix());              
            }

            br.close();
            fr.close();
        }
        catch(Exception e)
        {
            System.out.println("ERROR: Loading network.");
            System.out.println(e);
        }
    }
}
