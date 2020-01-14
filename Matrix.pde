import java.util.ArrayList;
import java.util.List;

public class Matrix
{
    private double[][] matrix;

    /**
     * Initializes random m x n matrix.
     * @param m     Row size of the matrix
     * @param n     Column size of the matrix
     */
    public Matrix(int m, int n)
    {
        //create matrix 
        matrix = new double[m][n];

        //initialize matrix to random values
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                matrix[i][j] = Math.random();
            }
        }
    }

    /**
     * Constructor for loading a matrix from a file.
     * @param values        Two dimensional array of the matrix values       
     */
    public Matrix(String[][] values)
    {
        int m = values.length;
        int n = values[0].length;

        matrix = new double[m][n];

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                matrix[i][j] = Double.parseDouble(values[i][j]);
            }
        }
    }

    /**
     * Constructor for loading a matrix from a file.
     * @param values        One dimensional array of the matrix values       
     */
    public Matrix(String[] values)
    {
        int n = values.length;

        matrix = new double[1][n];

        for (int i = 0; i < n; i++)
        {
            matrix[0][i] = Double.parseDouble(values[i]);
        }
    }

    /**
     * Constructor for a specified number of columns. Fills with random numbers
     * @param cols  Number of columns
     */
    public Matrix(int cols)
    {
        matrix = new double[1][cols];

        for (int i = 0; i < cols; i++)
        {
            matrix[0][i] = Math.random();
        }
    }

    /**
     * @param oneDimArray   One dimensional array to be converted to a matrix
     */
    public Matrix(double[] oneDimArray)
    {
        int n = oneDimArray.length;

        matrix = new double[1][n];

        for (int i = 0; i < n; i++)
        {
            matrix[0][i] = oneDimArray[i];
        }
    }

    /**
     * @param twoDimArray   Two dimensional array to be converted to a matrix
     */
    public Matrix(double[][] twoDimArray)
    {
        int m = twoDimArray.length;
        int n = twoDimArray[0].length;
        matrix = new double[m][n];

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                matrix[i][j] = twoDimArray[i][j];
            }
        }
    }

    /**
     * Empty constructor used for initializing empty Matrix instances.
     */
    public Matrix()
    {
        return;
    }

    /**
     * Method that performs matrix multiplication between matrix A and B
     * @param A     Matrix A
     * @param B     Matrix B
     * @return      Resultant Matrix of the multiplication between matrix A and B.
     */
    public Matrix MatrixMultiply(Matrix A, Matrix B)
    {
        double[][] matrixA = A.getWeights();
        double[][] matrixB = B.getWeights();

        // (mA x nA) x (mB x nB)
        int mA = matrixA.length;
        int nA = matrixA[0].length;

        int mB = matrixB.length;
        int nB = matrixB[0].length;


        //check nA = mB
        if (nA != mB)
        {
            System.out.println("Error! Invalid Matrix Dimensions!");
            System.exit(0);
        }

        double[][] result = new double[mA][nB];

        //multiplying matrices...
        for (int i = 0; i < mA; i++)
        {
            for (int j = 0; j < nB; j++)
            {
                for (int k = 0; k < nA; k++)
                {
                    result[i][j] += matrixA[i][k] * matrixB[k][j];
                }
            }
        }

        //return result as a new matrix
        return new Matrix(result);
    }

    /**
     * Method to perform element-wise multiplication between two Matrices. E.g. A[0] * B[0], A[1] * B[1], ..., A[n] * B[n], etc.
     * @param A     Matrix A
     * @param B     Matrix B
     * @return      Resultant Matrix of the element-wise multiplication of the two matrices.
     */
    public Matrix ElementWiseMultiply(Matrix A, Matrix B)
    {
        double[][] matrixA = A.getWeights();
        double[][] matrixB = B.getWeights();

        int mA = matrixA.length;
        int nA = matrixA[0].length;

        int mB = matrixB.length;
        int nB = matrixB[0].length;

        if (mA != mB || nA != nB)
        {
            System.out.println("ERROR: ElementWiseMultiply Dimensions Don't Match");
            System.exit(-1);
        }

        double[][] result = new double[mA][nA];

        for (int i = 0; i < mA; i++)
        {
            for (int j = 0; j < nA; j++)
            {
                result[i][j] = matrixA[i][j] * matrixB[i][j];
            }
        }
        
        return new Matrix(result);
    }

    /**
     * Method to perform element-wise addition between two matrices
     * @param A     Matrix A
     * @param B     Matrix B
     * @return      Resultant Matrix of the element-wise addition of the two matrices.
     */
    public Matrix ElementWiseAddition(Matrix A, Matrix B)
    {
        double[][] matrixA = A.getWeights();
        double[][] matrixB = B.getWeights();

        int mA = matrixA.length;
        int nA = matrixA[0].length;

        int mB = matrixB.length;
        int nB = matrixB[0].length;

        if (mA != mB || nA != nB)
        {
            System.out.println("ERROR: ElementWiseAddition Dimensions Don't Match");
            System.exit(0);
        }

        double[][] result = new double[mA][nA];

        for (int i = 0; i < mA; i++)
        {
            for (int j = 0; j < nA; j++)
            {
                result[i][j] = matrixA[i][j] + matrixB[i][j];
            }
        }
        
        return new Matrix(result);
    }

    /**
     * Method to perform element-wise subtraction between two matrices
     * @param A     Matrix A
     * @param B     Matrix B
     * @return      Resultant Matrix of the element-wise subtraction of the two matrices.
     */
    public Matrix ElementWiseSubtraction(Matrix A, Matrix B)
    {
        double[][] matrixA = A.getWeights();
        double[][] matrixB = B.getWeights();

        int mA = matrixA.length;
        int nA = matrixA[0].length;

        int mB = matrixB.length;
        int nB = matrixB[0].length;

        if (mA != mB || nA != nB)
        {
            System.out.println("ERROR: ElementWiseSubtraction Dimensions Don't Match");
            System.exit(-1);
        }

        double[][] result = new double[mA][nA];

        for (int i = 0; i < mA; i++)
        {
            for (int j = 0; j < nA; j++)
            {
                result[i][j] = matrixA[i][j] - matrixB[i][j];
            }
        }
        
        return new Matrix(result);
    }


    /**
     * Method to transpose a Matrix
     * @return      The transpose of the current matrix
     */
    public Matrix transpose()
    {
        double[][] result = new double[matrix[0].length][matrix.length];

        for (int i = 0; i < matrix[0].length; i++)
        {
            for (int j = 0; j < matrix.length; j++)
            {
                result[i][j] = matrix[j][i];
            }
        }

        return new Matrix(result);
    }

    /**
     * Creates a copy of the current matrix
     * @return      A new copy of the current matrix.
     */
    public Matrix copyMatrix()
    {
        Matrix copy = new Matrix(matrix.length, matrix[0].length);

        for (int i = 0; i < matrix.length; i++)
        {
            for (int j = 0; j < matrix[0].length; j++)
            {
                copy.setValue(i, j, matrix[i][j]);
            }
        }

        return copy;
    }

    /**
     * Method to set a value in the matrix
     * @param row       row index
     * @param col       column index
     * @param value     value to be set
     */
    public void setValue(int row, int col, double value)
    {
        matrix[row][col] = value;
    }

    /**
     * Method for the Neural Network class. The values of the Matrix are considered to be weights in this case.
     */
    public double[][] getWeights()
    {
        return matrix;
    }

    /**
     * @param A         The Matrix to be scaled
     * @param factor    The factor by which each element will be multiplied.
     */
    public Matrix scaleMatrix(Matrix A, double factor)
    {
        double[][] values = A.getWeights();
        int m = values.length;
        int n = values[0].length;
        double[][] result = new double[m][n];

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                result[i][j] = values[i][j] * factor;
            }
        }

        return new Matrix(result);
    }

    /**
     * Method to return the sum of all elements in the matrix
     */
    public double sumMatrix()
    {
        double sum = 0;
        double[][] weights = getWeights();
        int m = weights.length;
        int n = weights[0].length;

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                sum += weights[i][j];
            }
        }

        return sum;
    }

    /**
     * 
     * Activation Function / Derivative Section:
     * 
     */

    /**
     * Method to determine if the activation function is valid or not.
     * Valid activation functions = sigmoid, tanh, linear, and, relu
     * @param activation_function       The name of the activation function to check.
     */
    public boolean validFunction(String activation_function)
    {
        if (!activation_function.equals("sigmoid") && !activation_function.equals("tanh") && !activation_function.equals("linear") && !activation_function.equals("relu"))
        {
            return false;
        }
        else
        {
            return true;
        }
    }
    
    /**
     * Method to find the correct activation function based off of the given string and then apply that 
     * function to the given Matrix.
     * 
     * @param activation_function       The activation function to apply to the Matrix.
     * @param A                         Matrix A that will have the function applied on.
     * @param Derivative              True if the derivative should be calculated, False if the function should be calculated.
     */
    public Matrix stringToFunctionMap(String activation_function, Matrix A, boolean Derivative)
    {
        Matrix answerMatrix = null;

        if (activation_function.equals("sigmoid"))
        {
            if (Derivative)
            {
                answerMatrix = sigmoid_derivative(A);
            }
            else
            {
                answerMatrix = sigmoid(A);
            }
        }
        else if (activation_function.equals("linear"))
        {
            if (Derivative)
            {
                answerMatrix = linear_derivative(A);
            }
            else
            {
                answerMatrix = linear(A);
            }
        }
        else if (activation_function.equals("tanh"))
        {
            if (Derivative)
            {
                answerMatrix = tanh_derivative(A);
            }
            else
            {
                answerMatrix = tanh(A);
            }
        }
        else if (activation_function.equals("relu"))
        {
            if (Derivative)
            {
                answerMatrix = relu_derivative(A);
            }
            else
            {
                answerMatrix = relu(A);
            }
        }
        else
        {
            System.out.println("ERROR: Activation Function not recognized!");
            System.exit(-1);
        }

        return answerMatrix;
    }

    /**
     * Sigmoid activation function
     * @param A     Matrix A that we will perform sigmoid activation function on
     * @return      A new copy of the Matrix after the sigmoid of each element has been calculated.
     */
    public Matrix sigmoid(Matrix A)
    {
        double[][] matrix = A.getWeights();
        double[][] result = new double[matrix.length][matrix[0].length];

        for (int i = 0; i < matrix.length; i++)
        {
            for (int j = 0; j < matrix[0].length; j++)
            {
                result[i][j] = sigmoid(matrix[i][j]);
            }
        }

        return new Matrix(result);
    }

    /**
     * Linear activation function
     * @param A     Matrix to be copied
     * @return      Matrix with linear activation function applied
     */
    public Matrix linear(Matrix A)
    {
        double[][] matrix = A.getWeights();

        return new Matrix(matrix);
    }

    /**
     * Hyperbolic tangent activation function.
     * @param A     The matrix that will have tanh applied to it.
     */
    public Matrix tanh(Matrix A)
    {
        double[][] matrix = A.getWeights();
        double[][] result = new double[matrix.length][matrix[0].length];

        for (int i = 0; i < matrix.length; i++)
        {
            for (int j = 0; j < matrix[0].length; j++)
            {
                result[i][j] = tanh(matrix[i][j]);
            }
        }

        return new Matrix(result);
    }

    /**
     * relu activation function
     * @param A     The matrix that will have relu applied to it, 
     */
    public Matrix relu(Matrix A)
    {
        double[][] matrix = A.getWeights();
        double[][] result = new double[matrix.length][matrix[0].length];

        for (int i = 0; i < matrix.length; i++)
        {
            for (int j = 0; j < matrix[0].length; j++)
            {
                result[i][j] = relu(matrix[i][j]);
            }
        }

        return new Matrix(result);
    }

    /**
     * Sigmoid function definition.
     * @param x     Value to be input into the simgoid function.
     * @return      Result of sigmoid(x)
     */
    public double sigmoid(double x)
    {
        return 1 / (1 + Math.exp(-x));
    }

    /**
     * Tanh function definition
     * @param x     Value to be input into the tanh function.
     * @return      Result of tanh(x)
     */
    public double tanh(double x)
    {
        return 2 / (1 + Math.exp(-2 * x)) - 1;
    }

    /**
     * relu function definition
     * @param x     Value to be input into the relu function
     * @return      Result of relu(x)
     */
    public double relu(double x)
    {
        if (x > 0)
        {
            return x;
        }
        else
        {
            return 0;
        }
    }

    /**
     * Method to calculate the derivative of the sigmoid function.
     * @param A     Matrix to be used as input
     * @return      The sigmoid_derivatives of every element in the Matrix
     */
    public Matrix sigmoid_derivative(Matrix A)
    {
        double[][] matrix = A.getWeights();
        double[][] result = new double[matrix.length][matrix[0].length];

        for (int i = 0; i < matrix.length; i++)
        {
            for (int j = 0; j < matrix[0].length; j++)
            {
                result[i][j] = sigmoid(matrix[i][j]) * (1 - sigmoid(matrix[i][j]));
            }
        }

        return new Matrix(result);
    }

    /**
     * Method to calculate the derivative of the linear activation function
     * @param A     The Matrix that will be used as input
     * @return      The linear function derivatives of every element in Matrix A.
     */
    public Matrix linear_derivative(Matrix A)
    {
        double[][] matrix = A.getWeights();
        double[][] result = new double[matrix.length][matrix[0].length];

        for (int i = 0; i < matrix.length; i++)
        {
            for (int j = 0; j < matrix[0].length; j++)
            {
                result[i][j] = 1;
            }
        }

        return new Matrix(result);
    }

    /**
     * Method to calculate the derivative of the tanh activation function
     * @param A     The Matrix that will be used as input
     * @return      The tanh derivative of every element in Matrix A
     */
    public Matrix tanh_derivative(Matrix A)
    {
        double[][] matrix = A.getWeights();
        double[][] result = new double[matrix.length][matrix[0].length];

        double tan_h;

        for (int i = 0; i < matrix.length; i++)
        {
            for (int j = 0; j < matrix[0].length; j++)
            {
                tan_h = tanh(result[i][j]);
                result[i][j] = 1 - tan_h * tan_h; 
            }
        }

        return new Matrix(result);
    }


    /**
     * Method to calculate the derivative of the relu activation function.
     * @param A     The matrix that will be used as input.
     * @return      The relu derivative of every element in A.
     */
    public Matrix relu_derivative(Matrix A)
    {
        double[][] matrix = A.getWeights();
        double[][] result = new double[matrix.length][matrix[0].length];

        for (int i = 0; i < matrix.length; i++)
        {
            for (int j = 0; j < matrix[0].length; j++)
            {
                if (matrix[i][j] > 0)
                {
                    result[i][j] = 1;
                }
                else
                {
                    result[i][j] = 0;
                }
            }
        }

        return new Matrix(result);
    }

    /**
     * toString Method to return information about the Matrix.
     */
    public String toString()
    {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < matrix.length; i++)
        {
            sb.append("[ ");
            for (int j = 0; j < matrix[0].length; j++)
                sb.append(matrix[i][j] + ", ");

            sb.append("]");

            if (i != matrix.length - 1)
                sb.append("\n");
        }

        sb.append("]");

        return sb.toString();
    }


}
