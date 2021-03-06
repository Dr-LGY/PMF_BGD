
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.HashMap;
import java.util.HashSet;

public class Data 
{
	// === Configurations	
	// the number of latent dimensions
	public static int d = 20; 

	// tradeoff $\alpha_u$
	public static float alpha_u = 0.01f;
	// tradeoff $\alpha_v$
	public static float alpha_v = 0.01f;
	 
	// learning rate $\gamma$
	public static float gamma = 1000f;
	
	public static float rho = 0.8f;

	 // === Input data files
	public static String fnTrainData = "C:\\Users\\LGY\\Desktop\\DATA\\ml-100k\\u1.base";
	public static String fnTestData = "C:\\Users\\LGY\\Desktop\\DATA\\ml-100k\\u1.test";
	public static String fnOutputData = "";

	// 
	public static int n = 943; // number of users
	public static int m = 1682; // number of items
	public static int num_train; // number of training triples of (user,item,rating)
	public static int num_test; // number of test triples of (user,item,rating)

	public static float MinRating = 1.0f; // minimum rating value (0.5 for ML10M, Flixter; 1 for Netflix)
	public static float MaxRating = 5.0f; // maximum rating value

	// scan number over the whole data
	public static int num_iterations = 500; //600

	// === training data (target data)
	public static float[][] ratings;

	// === test data
	public static int[] indexUserTest;
	public static int[] indexItemTest;
	public static float[] ratingTest;

	// === model parameters to learn, start from index "1"
	public static float[][] U;
	public static float[][] V;

	// === file operation
	public static FileWriter fw ;
	public static BufferedWriter bw;
}
