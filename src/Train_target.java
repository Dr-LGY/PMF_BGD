import java.io.IOException;

public class Train_target {
	public static void train(int iterations) {
		//long startTime=System.nanoTime();
		for (int iter = 0; iter < iterations; iter++){
			
			System.out.print("Iter:" + Integer.toString(iter) + "| ");
			
			try {
				Test.test();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
	
			double one_div_NM =  1.0 / (Data.n * Data.m);
			for (int u = 1; u <= Data.n; u++) {
				float grad_U[] = new float[Data.d];
				for (int i = 1; i <= Data.m; i++ ) {
					if (Data.r[u][i] != 0) {
						float pred = 0; 
						for (int f=0; f<Data.d; f++)
						{
							pred += Data.U[u][f] * Data.V[i][f];
						}
						float error = Data.r[u][i] - pred;
						for(int f=0; f<Data.d; f++)
						{	
							
							grad_U[f] += -error * Data.V[i][f] + Data.alpha_u * Data.U[u][f];
								    			
						}
					}
				}
				
				for(int f=0; f<Data.d; f++) Data.U[u][f] = (float) (Data.U[u][f] - Data.gamma * grad_U[f] * one_div_NM);
			}
			
			for (int i = 1; i <= Data.m; i++) {
				float grad_V[] = new float[Data.d];
				for (int u = 1; u <= Data.n; u++) {
					if (Data.r[u][i] != 0) {
						float pred = 0;
						for (int f=0; f<Data.d; f++)
						{
							pred += Data.U[u][f] * Data.V[i][f];
						}
						float error = Data.r[u][i] - pred;
						for(int f=0; f<Data.d; f++)
						{	
							grad_V[f] += -error * Data.U[u][f]   + Data.alpha_v * Data.V[i][f];
							
								    			
						}
					}
				}
				for(int f=0; f<Data.d; f++) Data.V[i][f] = (float) (Data.V[i][f] - Data.gamma * grad_V[f] * one_div_NM);
			}
			Data.gamma = (float) (Data.gamma * ((float)(200 * iterations) / (iter + 200 * iterations)) );
		} 	
		//long endTime=System.nanoTime(); 

	//	System.out.println("程序运行时间： "+(endTime-startTime)+"ns");
	}
	
}
