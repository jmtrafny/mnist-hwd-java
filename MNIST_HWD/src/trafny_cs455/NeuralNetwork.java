package trafny_cs455;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.Random;

public class NeuralNetwork {
	Random xd = new Random();
	String fileName = new SimpleDateFormat("'Results_'hh-mm-ss'.txt'").format(new Date());

	private int input_nodes;
	private int hidden_nodes;
	private int output_nodes;
	private int[] input_vals;
	private double h_threshold;
	private double o_threshold;
	private double[] hidden_vals;
	private double[] output_vals;
	private double[] error_signal;
	private double[] o_error_gradient;
	private double[] h_error_gradient;
	private double[][] hidden_weights;
	private double[][] output_weights;

	NeuralNetwork(int[] sizes) {
		input_nodes = sizes[0];
		hidden_nodes = sizes[1];
		output_nodes = sizes[2];

		hidden_weights = new double[input_nodes][hidden_nodes];
		output_weights = new double[hidden_nodes][output_nodes];

		// Randomly initialize weights and threshold
		h_threshold = -0.5 + xd.nextDouble();
		o_threshold = -0.5 + xd.nextDouble();
		for (int i = 0; i < input_nodes; i++) {
			for (int h = 0; h < hidden_nodes; h++) {
				hidden_weights[i][h] = -0.5 + xd.nextDouble();
			}
		}
		for (int h = 0; h < hidden_nodes; h++) {
			for (int o = 0; o < output_nodes; o++) {
				output_weights[h][o] = -0.5 + xd.nextDouble();
			}
		}
	}

	// Sigmoid activation function
	double sigmoid(double val) {
		return 1 / (1 + Math.exp(-1 * val));
	}

	void train(List<int[][]> train_images, int[] train_labels, double learningRate) throws IOException {
		int count = 0;
		int numRight = 0;

		// Prepare output file
		//
		FileWriter fileWriter = new FileWriter("results/Training_" + fileName);
		PrintWriter printWriter = new PrintWriter(fileWriter);

		// Training loop
		// Do this for each image in our training set
		//
		for (int[][] image : train_images) {
			// Get answer and encode it
			int answer = train_labels[count];
			int[] encodedAnswer = new int[10];
			encodedAnswer[answer] = 1;

			// Get input values by flattening the 2d image array into
			// a 1d array of input values to act as our input 'nodes'
			//
			input_vals = new int[image.length * image[0].length];
			int index = 0;
			for (int r = 0; r < image.length; r++) {
				for (int c = 0; c < image[0].length; c++) {
					input_vals[index] = image[r][c];
					index++;
				}
			}

			// Calculate hidden values
			// summation(input * weight - threshold)
			//
			hidden_vals = new double[hidden_nodes];
			for (int h = 0; h < hidden_vals.length; h++) {
				double sum = 0;
				for (int i = 0; i < input_vals.length; i++) {
					sum += input_vals[i] * hidden_weights[i][h] - h_threshold;
				}
				// Activate finished neuron
				hidden_vals[h] = sigmoid(sum);
			}

			// Calculate output values
			// This is done in the same manner as the hidden values!
			//
			output_vals = new double[output_nodes];
			for (int o = 0; o < output_vals.length; o++) {
				double sum = 0;
				for (int h = 0; h < hidden_vals.length; h++) {
					sum += hidden_vals[h] * output_weights[h][o] - o_threshold;
				}
				// Activate finished neuron
				output_vals[o] = sigmoid(sum);
			}

			// Is our guess correct?
			//
			int guess = 0;
			for (int i = 0; i < output_vals.length; i++) {
				guess = output_vals[i] > output_vals[guess] ? i : guess;
			}
			if (guess == answer) {
				numRight++;
			}

			// Output weight training - Backpropigation
			// Pg 178 of Textbook (Pg 197 of PDF version)
			//
			error_signal = new double[output_nodes];
			o_error_gradient = new double[output_nodes];
			// Calculate error signal and gradient
			for (int o = 0; o < output_vals.length; o++) {
				error_signal[o] = encodedAnswer[o] - output_vals[o];
				o_error_gradient[o] = sigmoid(output_vals[o]) * (1 - sigmoid(output_vals[o])) * error_signal[o];
				// Calculate the weight corrections
				for (int h = 0; h < hidden_nodes; h++) {
					double correction = learningRate * sigmoid(hidden_vals[h]) * o_error_gradient[o];
					output_weights[h][o] = output_weights[h][o] + correction;
				}
			}

			// Hidden weight training - Backpropigation
			// Different calculations that output training!!!
			//
			h_error_gradient = new double[hidden_nodes];
			for (int h = 0; h < hidden_vals.length; h++) {
				double summation = 0;
				for (int o = 0; o < output_nodes; o++) {
					summation += o_error_gradient[o] * output_weights[h][o];
				}
				h_error_gradient[h] = sigmoid(hidden_vals[h]) * (1 - sigmoid(hidden_vals[h])) * summation;
				// Calculate hidden weight corrections
				for (int i = 0; i < input_nodes; i++) {
					double correction = learningRate * input_vals[i] * h_error_gradient[h];
					hidden_weights[i][h] = hidden_weights[i][h] + correction;
				}
			}

			// Finished training on this image
			// prepare epoc results
			//
			count++;
			if (count % 1000 == 0) {
				System.out.printf("Training on image %d \n", count);
				double percentCorrect = ((double) numRight / count) * 100;
				System.out.printf("\tCorrect/Total --> %d/%d = %2.3f \n", numRight, count, percentCorrect);

				// Write to File
				//
				printWriter.printf("\tCorrect/Total --> %d/%d = %2.3f \n", numRight, count, percentCorrect);
			}
		} // rof

		System.out.println("\tTraining Complete!");
		printWriter.close();
	}

	void test(List<int[][]> test_images, int[] test_labels) throws IOException {

		int numRight = 0;
		int numTotal = test_images.size();
		double percentCorrect;
		int count = 0;

		// Prepare output file
		//
		FileWriter fileWriter = new FileWriter("results/Testing_" + fileName);
		PrintWriter printWriter = new PrintWriter(fileWriter);

		for (int[][] image : test_images) {
			// Get answer and encode it
			int answer = test_labels[count];
			int[] encodedAnswer = new int[10];
			encodedAnswer[answer] = 1;

			// Get input values
			input_vals = new int[image.length * image[0].length];
			int index = 0;
			for (int r = 0; r < image.length; r++) {
				for (int c = 0; c < image[0].length; c++) {
					input_vals[index] = image[r][c];
					index++;
				}
			}

			// Calculate hidden values
			hidden_vals = new double[hidden_nodes];
			for (int h = 0; h < hidden_vals.length; h++) {
				double sum = 0;
				for (int i = 0; i < input_vals.length; i++) {
					// hidden_vals[h] += input_vals[i] * hidden_weights[i][h];
					sum += input_vals[i] * hidden_weights[i][h] - h_threshold;
				}
				// Activate finished neuron
				hidden_vals[h] = sigmoid(sum);
			}

			// Calculate output values
			output_vals = new double[output_nodes];
			for (int o = 0; o < output_vals.length; o++) {
				double sum = 0;
				for (int h = 0; h < hidden_vals.length; h++) {
					// output_vals[o] += hidden_vals[h] * output_weights[h][o];
					sum += hidden_vals[h] * output_weights[h][o] - o_threshold;
				}
				// Activate finished neuron
				output_vals[o] = sigmoid(sum);
			}

			// Compare to real value
			int guess = 0;
			for (int i = 0; i < output_vals.length; i++) {
				guess = output_vals[i] > output_vals[guess] ? i : guess;
			}

			if (guess == answer) {
				numRight++;
			}

			System.out.printf("Guess -> %d %d <- Actual\n", guess, answer);
			count++;
		}
		percentCorrect = ((double) numRight / numTotal) * 100;
		System.out.printf("\tComplete\nCorrect/Total --> %d/%d = %2.3f", numRight, numTotal, percentCorrect);
		printWriter.printf("\tCorrect/Total --> %d/%d = %2.3f \n", numRight, numTotal, percentCorrect);
		printWriter.close();
	}

	void dump() throws IOException {

		// Prepare output file
		//
		FileWriter fileWriter = new FileWriter("results/META_" + fileName);
		PrintWriter printWriter = new PrintWriter(fileWriter);

		// private int input_nodes
		// # of input nodes
		//
		printWriter.printf("Number of Input Nodes: %d\n", input_nodes);

		// private int hidden_nodes
		// number of hidden nodes
		//
		printWriter.printf("Number of Hidden Nodes: %d\n", hidden_nodes);

		// private int output_nodes;
		// number of output nodes
		//
		printWriter.printf("Number of Output Nodes: %d\n", output_nodes);

		// private double h_threshold;
		// randomly initalized threshold value for hidden layer
		//
		printWriter.printf("Threshold value for hidden layer: %f\n", h_threshold);

		// private double o_threshold;
		// randomly initalized threshold value for output layer
		//
		printWriter.printf("Threshold value for output layer: %f\n", o_threshold);

		// private double[][] hidden_weights;
		// 2d array representing the weights from input to hidden layer
		// row == input layer
		// col == hidden layer
		// hidden_weights[row][col] = the weight value between them
		printWriter.printf("Hidden Layer Weights:\n");
		printWriter.println(
				Arrays.deepToString(hidden_weights).replace("], ", "]\n").replace("[[", "[").replace("]]", "]"));

		// private double[][] output_weights;
		// 2d array representing the weights from hidden to output layer
		// row == hidden layer
		// col == output layer
		// hidden_weights[row][col] = the weight value between them
		printWriter.printf("Output Layer Weights:\n");
		printWriter.println(
				Arrays.deepToString(output_weights).replace("], ", "]\n").replace("[[", "[").replace("]]", "]"));
		
		// Finished!
		//
		printWriter.close();
		

	}

}
