package trafny_cs455;

import java.io.IOException;
import java.util.List;
import java.util.Scanner;

public class Main {

	private static final double learningRate = 0.1;

	public static void main(String[] args) {
		Scanner police = new Scanner(System.in);
		System.out.print("Preparing Data ");

		// Preprocess the MNIST dataset and prepare it for the ANN.
		//
		String TRAIN_IMAGE_FILE = "data/train-images.idx3-ubyte";
		String TRAIN_LABEL_FILE = "data/train-labels.idx1-ubyte";
		String TEST_IMAGE_FILE = "data/t10k-images.idx3-ubyte";
		String TEST_LABEL_FILE = "data/t10k-labels.idx1-ubyte";

		List<int[][]> train_images = MnistReader.getImages(TRAIN_IMAGE_FILE);
		int[] train_labels = MnistReader.getLabels(TRAIN_LABEL_FILE);
		List<int[][]> test_images = MnistReader.getImages(TEST_IMAGE_FILE);
		int[] test_labels = MnistReader.getLabels(TEST_LABEL_FILE);

		int inputLayerSize = train_images.get(0).length * train_images.get(0)[0].length;
		int hiddenLayerSize = 56; // 300.. 28... Who knows what's best...
		int outputLayerSize = 10; // 0-9
		int[] sizes = { inputLayerSize, hiddenLayerSize, outputLayerSize };

		NeuralNetwork n = new NeuralNetwork(sizes);

		// Ready to train network
		// Wait for user to be ready
		//
		System.out.print(" - Complete\n");
		System.out.print("READY TO TRAIN -- PRESS ENTER TO BEGIN");
		police.nextLine();

		// Instanciate the NN with the sizes of each layer
		//
		try {
			n.train(train_images, train_labels, learningRate);
		} catch (IOException e) {
			// Could not create file
			e.printStackTrace();
		}

		// Ready to test network
		// Wait for user to be ready
		//
		System.out.println("READY TO TEST -- PRESS ENTER TO BEGIN");
		police.nextLine();

		// Test the network!
		//
		try {
			n.test(test_images, test_labels);
		} catch (IOException e) {
			// Could not create file
			e.printStackTrace();
		}

		// Dump the metadata!
		//
		try {
			n.dump();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		// All done!
		//
		police.close();
	}
}
