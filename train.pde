
int[][] grid;
int[] playerLoc;
int[][] enemyLoc;
int gridSize;
int score;
boolean gameOver;
boolean won;
int NUM_EPOCHS = 7500;
double GAMMA = 0.8;
NeuralNetwork nn;
ArrayList<String> path;
PrintWriter outputWriter;
boolean startDraw;

void setup() {
    outputWriter = createWriter("scores.txt");
    size(500, 500);
    startDraw = false;
    noLoop();
    
    initializeGame();
    
    //Initialize Neural Network
    nn = new NeuralNetwork(0.0001, 100000000);
    nn.addInputLayer(4,5, "relu");
    nn.addLayer(4, "linear");

    //perform training
    training();
    
    outputWriter.flush();
    outputWriter.close();
    
    loop();
    
    System.out.println("Testing Agent:");
    startDraw = true;
    
}

void initializeGame()
{
    score = 0;
    gridSize = 5;
    gameOver = false;
    won = false;
    grid = new int[gridSize][gridSize];
    grid[gridSize / 2][gridSize / 2] = 2;//player
    path = new ArrayList<String>();
    playerLoc = new int[2];
    playerLoc[0] = gridSize / 2;
    playerLoc[1] = gridSize / 2;
    
    //enemies
    enemyLoc = new int[6][2];
    enemyLoc[0][0] = 0;
    enemyLoc[0][1] = 0;
    
    enemyLoc[1][0] = gridSize - 1;
    enemyLoc[1][1] = 0;
    
    enemyLoc[2][0] = gridSize - 1;
    enemyLoc[2][1] = gridSize - 1;
    
    enemyLoc[3][0] = 0;
    enemyLoc[3][1] = gridSize - 1;
    
    enemyLoc[4][0] = gridSize / 2 - 1;
    enemyLoc[4][1] = gridSize / 2;
    
    enemyLoc[5][0] = gridSize / 2 + 1;
    enemyLoc[5][1] = gridSize / 2;
}

void training() {
    for (int i = 1; i <= NUM_EPOCHS; i++) {
        path = new ArrayList<String>();
        
        if (i % 1000 == 0) {
            System.out.printf("Epoch: %d\n", i);   
        }
        
        while (!gameOver && !won) {
            think(nn);
        }
        
        outputWriter.println(score);
        reset();
    }
    
    /* Optional printing of last path.
    System.out.println(path.size());
    for (int i = 0; i < path.size(); i++) {
        System.out.println(path.get(i));   
    } 
    */
}

void actionMapper(int index, boolean doneTraining) {
    if (index == 0) {
        path.add("LEFT");
        action('a', doneTraining);
    }
    else if (index == 1) {
        path.add("DOWN");
        action('s', doneTraining);   
    }
    else if (index == 2) {
        path.add("RIGHT");
        action('d', doneTraining);   
    }
    else if (index == 3) {
        path.add("UP");
        action('w', doneTraining);   
    }
}

int[] features() {
    int[] f = new int[4];
    
    int up = Integer.MAX_VALUE;
    int down = Integer.MAX_VALUE;
    int right = Integer.MAX_VALUE;
    int left = Integer.MAX_VALUE;
    
    for (int i = 0; i < enemyLoc.length; i++) {
        if (enemyLoc[i][1] == playerLoc[1]) {
            if (enemyLoc[i][0] < playerLoc[0]) {
                up = Math.min(up, playerLoc[0] - enemyLoc[i][0]);   
            }
            else
            {
                up = Math.min(up, playerLoc[0] + (gridSize - 1) - enemyLoc[i][0]);
            }
            
            if (enemyLoc[i][0] > playerLoc[0]) {
                down = Math.min(down, enemyLoc[i][0] - playerLoc[0]);   
            }
            else {
                down = Math.min(down, gridSize - playerLoc[0] + enemyLoc[i][0]);
            }
        }
        
        if (enemyLoc[i][0] == playerLoc[0]) {
            if (enemyLoc[i][1] > playerLoc[1]) {
                right = Math.min(right, enemyLoc[i][1] - playerLoc[1]);
            }
            else {
                right = Math.min(right, gridSize - playerLoc[1] + enemyLoc[i][1]);   
            }
            
            if (playerLoc[1] > enemyLoc[i][1]) {
                 left = Math.min(left, playerLoc[1] - enemyLoc[i][1]);   
            }
            else {
                left = Math.min(left, playerLoc[1] + (gridSize - 1) - enemyLoc[i][1]); 
            }
        }
        
    }
    
    if (up == Integer.MAX_VALUE){
        up = 0;
    }
    
    if (down == Integer.MAX_VALUE) {
        down = 0;
    }
    
    if (left == Integer.MAX_VALUE) {
        left = 0;
    }
    
    if (right == Integer.MAX_VALUE) {
        right = 0;
    }


    f[0] = up;
    f[1] = down;
    f[2] = left;
    f[3] = right;
    
    return f;
}

void reset() {
    size(500, 500);
    score = 0;
    gameOver = false;
    won = false;
    gridSize = 5;
    grid = new int[gridSize][gridSize];
    grid[gridSize / 2][gridSize / 2] = 2;//player
    
    playerLoc = new int[2];
    playerLoc[0] = gridSize / 2;
    playerLoc[1] = gridSize / 2;
    
    //enemies
    enemyLoc = new int[6][2];
    enemyLoc[0][0] = 0;
    enemyLoc[0][1] = 0;
    
    enemyLoc[1][0] = gridSize - 1;
    enemyLoc[1][1] = 0;
    
    enemyLoc[2][0] = gridSize - 1;
    enemyLoc[2][1] = gridSize - 1;
    
    enemyLoc[3][0] = 0;
    enemyLoc[3][1] = gridSize - 1;
    
    enemyLoc[4][0] = gridSize / 2 - 1;
    enemyLoc[4][1] = gridSize / 2;
    
    enemyLoc[5][0] = gridSize / 2 + 1;
    enemyLoc[5][1] = gridSize / 2;
}


void draw() {
    int size = 500 / 5;
    fill(255,255,255);
    
    for (int i = 0; i < grid.length; i++) {
        for (int j = 0; j < grid[0].length; j++) {
             rect(size * j, size * i, size, size);
        }
    }
    
    fill(255,0,0);
    for (int i = 0; i < enemyLoc.length; i++) {
        rect(enemyLoc[i][1] * size, enemyLoc[i][0] * size, size, size);
    }
    
    fill(0, 0, 255);
    rect(playerLoc[1] * size, playerLoc[0] * size, size, size);
    
    if (startDraw) {
        demoNetwork();
        delay(500);
    }
}

void demoNetwork() {
    double[] state = oneDimToTwoDim(features());
    nn.FeedForward(state, false);   
    Matrix outputLayer = nn.layers.get(nn.layers.size() - 1);
    double[][] outputValues = outputLayer.getWeights();
    
    int maxInd = valueLearningIndex(outputValues);
    /* Optional data to view relating to the networks q value predictions.
    System.out.println(outputLayer);
    System.out.printf("maxInd = %d\n", maxInd);
    */
    actionMapper(maxInd, true);
}

void updateEnemies() {
    for (int i = 0; i < enemyLoc.length; i++) {
        enemyLoc[i][1] += 1;
        if (enemyLoc[i][1] == grid[0].length) {
            enemyLoc[i][1] = 0;
            enemyLoc[i][0] = (enemyLoc[i][0] + 1) % grid.length;
        }
    }
}

void keyPressed() {
    if (key == 'a'){
           playerLoc[1]--;
           
           if (playerLoc[1] == -1) {
               playerLoc[1] = gridSize - 1;   
           }
    }
    else if (key == 'd') {
        playerLoc[1]++;
        
        if (playerLoc[1] == gridSize) {
            playerLoc[1] = 0;   
        }
    }
    else if (key == 's') {
        playerLoc[0]++;
        
        if (playerLoc[0] == gridSize) {
            playerLoc[0] = 0;   
        }
    }
    else if (key == 'w') {
        playerLoc[0]--;
        
        if (playerLoc[0] == -1) {
            playerLoc[0] = gridSize - 1;   
        }
    }
    
    updateEnemies();

    int[] f = features();
    
    for (int i = 0; i < f.length; i++) {
        System.out.printf("%d, ", f[i]);
    }
    System.out.println();
    
    if (collisionCheck()){
        System.out.println("Score = " + score);
        gameOver = true;
        reset();
    }
    else
    {
        score++;   
        
        if (score == 30) {
            //System.out.println("Won Game");
            System.out.println("Score = " + score);
            
            won = true;
            reset();
        }
    }
    
    double[] state = oneDimToTwoDim(features());
    
    //calculate q-values:
    nn.FeedForward(state, false);
    
    //get output layer
    Matrix outputLayer = nn.layers.get(nn.layers.size() - 1);
    System.out.println(outputLayer);
}

void action(Character k, boolean doneTraining) {
       if (k == 'a'){
           playerLoc[1]--;
           
           if (playerLoc[1] == -1) {
               playerLoc[1] = gridSize - 1;   
           }
    }
    else if (k == 'd') {
        playerLoc[1]++;
        
        if (playerLoc[1] == gridSize) {
            playerLoc[1] = 0;   
        }
    }
    else if (k == 's') {
        playerLoc[0]++;
        
        if (playerLoc[0] == gridSize) {
            playerLoc[0] = 0;   
        }
    }
    else if (k == 'w') {
        playerLoc[0]--;
        
        if (playerLoc[0] == -1) {
            playerLoc[0] = gridSize - 1;   
        }
    }
    
    updateEnemies();
    
    if (collisionCheck()){
        if (!doneTraining) {
            gameOver = true;
        }
        else {
            System.out.println("Resetting");
            gameOver = true;
            reset();
        }
    }
    else
    {
        score++;   
        
        if (score == 30) {
            won = true;
            
            if (doneTraining) {
                System.out.println("Agent Won!");   
                reset();
            }
        }
    }
}

boolean collisionCheck() {
    for (int i = 0; i < enemyLoc.length; i++) {
        if (enemyLoc[i][0] == playerLoc[0] && enemyLoc[i][1] == playerLoc[1]) {
            return true;   
        }
    }
    
    return false;
}
