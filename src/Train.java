import java.io.IOException;


public class Train {
	
	public static void train(int iterations) {

		for (int iter = 0; iter < iterations; iter++){

//			if(iter % 10 == 0) // ÿ10��ѭ���������̨��ʾһ��
			{
				// output each iteration result
				try {
					Data.bw.write("Iter:" + Integer.toString(iter) + "| ");
					Data.bw.flush();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				
				System.out.print("Iter:" + Integer.toString(iter) + "| ");

				try {
					Test.test();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}

			}

			//compute gradient U of each user: grad_U
			float grad_U[][] = new float[Data.n + 1][Data.d];
			double one_div_NM =  1.0 / (Data.n * Data.m);
			
			for (int u = 1; u <= Data.n; u++) {

				for (int i = 1; i <= Data.m; i++ ) {
					if (Data.ratings[u][i] != 0) {
						float pred = 0; 
						for (int f=0; f<Data.d; f++)
						{
							pred += Data.U[u][f] * Data.V[i][f];
						}
						
						float error = Data.ratings[u][i] - pred;
						for(int f=0; f<Data.d; f++)
						{	
							grad_U[u][f] += -error * Data.V[i][f] + Data.alpha_u * Data.U[u][f];
						}
						
					}
				}

			}

			//compute gradient V of each item: grad_V
			float grad_V[][] = new float[Data.m + 1][Data.d];
			for (int i = 1; i <= Data.m; i++) {

				for (int u = 1; u <= Data.n; u++) {
					if (Data.ratings[u][i] != 0) {
						float pred = 0;
						for (int f=0; f<Data.d; f++)
						{
							pred += Data.U[u][f] * Data.V[i][f];
						}
						
						float error = Data.ratings[u][i] - pred;
						for(int f=0; f<Data.d; f++)
						{	
							grad_V[i][f] += -error * Data.U[u][f]   + Data.alpha_v * Data.V[i][f];
						}
												
					}
				}
			}

			//use grand_U and grad_V to update U and V matrix
			for (int u = 1; u <= Data.n; u++) {
				for(int f=0; f<Data.d; f++) 
				{
					Data.U[u][f] = (float) (Data.U[u][f] - Data.gamma * grad_U[u][f] * one_div_NM);
				}
			}

			for (int i = 1; i <= Data.m; i++) {
				for(int f=0; f<Data.d; f++) 
				{
					Data.V[i][f] = (float) (Data.V[i][f] - Data.gamma * grad_V[i][f] * one_div_NM);
				}
			}

			Data.gamma = (float) (Data.gamma * ((float)(200 * iterations) / (iter + 200 * iterations)) );
		} 	
	
	}

}