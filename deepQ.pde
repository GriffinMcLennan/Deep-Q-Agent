void think(NeuralNetwork nn) {
    //get current state:
    double[] state = oneDimToTwoDim(features());
    
    //calculate q-values:
    nn.FeedForward(state, false);
    
    //get output layer
    Matrix outputLayer = nn.layers.get(nn.layers.size() - 1);
    double[][] outputValues = outputLayer.getWeights();
    double[][] y_preds = new double[1][4];
    
    //Find action with maximum q-value:
    int maxInd = policyLearningIndex(outputValues);
    
    for (int i = 0; i < outputValues[0].length; i++) {
        y_preds[0][i] = outputValues[0][i];
    }
    
    //take that action
    actionMapper(maxInd, false);
    
    //get new state
    double[] newState = oneDimToTwoDim(features());
    
    //calculate reward:
    double reward;
    
    if (won) {
        reward = 10;  
    }
    else if (gameOver) {
        reward = -10;
    }
    else {
        reward = .5;
    }
    
    //calculate new q values
    nn.FeedForward(newState, false);
    
    outputLayer = nn.layers.get(nn.layers.size() - 1);
    outputValues = outputLayer.getWeights();
    
    //find maximum new q value:
    double newMax = outputValues[0][0];
    
    for (int i = 1; i < outputValues[0].length; i++) {
        if (newMax < outputValues[0][i]){
            newMax = outputValues[0][i];
        }
    }
   
    //gradient descent:
    //need to backpropagate that cost through the network for the old input
    nn.FeedForward(state, false); //feedforward old state.
    
    //Set the target for the prediction as the new q-value we found from traversing to the next state
    y_preds[0][maxInd] = reward + GAMMA * newMax;
    
    
    // [predict, predict, ..., maxInd: reward + newMax, predict, predict, ...]
    nn.train(oneDimToTwoDim(state), y_preds, 1, "sgd"); 
}

int valueLearningIndex(double[][] outputValues) {
    //Find action with maximum q-value:
    double max = outputValues[0][0];
    int maxInd = 0;
    
    for (int i = 1; i < outputValues[0].length; i++) {
        if (max < outputValues[0][i]){
            max = outputValues[0][i];
            maxInd = i;
        }
    }
    
    return maxInd; 
}

int policyLearningIndex(double[][] outputValues) {
    double[] normalized = new double[outputValues[0].length];
    int ind = 0;
    int expSum = 0;
    
    for (int i = 0; i < outputValues[0].length; i++) {
        expSum += Math.exp(outputValues[0][i]);   
    }
    
    for (int i = 0; i < normalized.length; i++) {
        normalized[i] = Math.exp(outputValues[0][i]) / expSum;
    }
    
    double rand = Math.random();
  
    while (ind < normalized.length - 1 && rand - normalized[ind] > 0) {
           rand = rand - normalized[ind];
           ind++;
    }
    
    return ind;
}
