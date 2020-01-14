double[][] oneDimToTwoDim(double[] arr) {
    double[][] ans = new double[1][arr.length];
    
    for (int i = 0; i < arr.length; i++) {
        ans[0][i] = arr[i];   
    }
    
    return ans;
}

double[] oneDimToTwoDim(int[] arr){
       double[] ret = new double[arr.length];
       
       for (int i = 0; i < arr.length; i++) {
           ret[i] = (double)arr[i];   
       }
       
       return ret;
}
