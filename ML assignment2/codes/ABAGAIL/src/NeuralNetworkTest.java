import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import func.nn.feedfwd.FeedForwardNeuralNetworkFactory;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;
import shared.DataSet;
import shared.ErrorMeasure;
import shared.Instance;
import shared.SumOfSquaresError;
import shared.filt.*;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.LineNumberReader;
import java.text.DecimalFormat;
import java.util.Scanner;

import shared.writer.MyWriter;
import util.linalg.Vector;

/**
 * Created by allam on 3/9/17.
 */
public class NeuralNetworkTest {
    //private static Instance[] instances;
    private static Instance[] trainInstances;
    private static Instance[] testInstances;

    private static int inputLayer = 10, hiddenLayer = 7, outputLayer = 1, trainingIterations = 6000;
    private static int testInterval = 10;

    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet trainSet, testSet;

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    private static String trainFile = "ABAGAIL/src/optdigits.tra";
    private static String testFile = "ABAGAIL/src/optdigits.tes";
    private static String trainFile1 = "ABAGAIL/src/magic04.data";
    private static String testFile1 = "ABAGAIL/src/magic04.data";
    private static String abaloneFile = "ABAGAIL/src/opt/test/abalone.txt";

    private static MyWriter RHCErrorWriter = new MyWriter("RHCError.csv", new String[]{"Iterations", "Error", "Train accuracy", "Test Accuracy", "time"});
    private static MyWriter SAErrorWriter = new MyWriter("SAError.csv", new String[]{"Iterations", "Error", "Train accuracy", "Test Accuracy", "time"});
    private static MyWriter GAErrorWriter = new MyWriter("GAError.csv", new String[]{"Iterations", "Error", "Train accuracy", "Test Accuracy", "time"});

    private static MyWriter[] errorWriters = new MyWriter[3];


    public static void main(String[] args) {
        trainInstances = readInstances(trainFile1);
        trainSet = new DataSet(trainInstances);
        RandomOrderFilter rof = new RandomOrderFilter();
        rof.filter(trainSet);
        TestTrainSplitFilter ttsf = new TestTrainSplitFilter(80);
        ttsf.filter(trainSet);
        trainSet = ttsf.getTrainingSet();
        testSet = ttsf.getTestingSet();
        trainInstances = trainSet.getInstances();
        testInstances = testSet.getInstances();
        testSA1();
        testGA();
        myTest();

    }


    private static Instance getOutputInstance(int t){
        double arr[] = new double[outputLayer];
        for (int i = 0; i < outputLayer; i++) arr[i] = 0;
        arr[t] = 1;
        return new Instance(arr);
    }

    private static int getLabel(Vector vec){
        int maxi = -1;
        double max = 0;
        for (int i =  0; i < vec.size(); i++){
            if (vec.get(i) > max){
                max = (vec.get(i));
                maxi = i;
            }
        }
        return maxi;
    }
    private static void testGA(){
        int t = 100;
        for (int i = 0; i < 6; i++) {
            BackPropagationNetwork network = factory.createClassificationNetwork(
                    new int[]{inputLayer, hiddenLayer, outputLayer});
            NeuralNetworkOptimizationProblem nnop = new NeuralNetworkOptimizationProblem(trainSet, network, measure);
            String fileName = "GA_test" + i + ".csv";
            MyWriter writer = new MyWriter(fileName, new String[]{"Iterations", "Error", "Train accuracy", "Test Accuracy", "time"});
            OptimizationAlgorithm oa = new StandardGeneticAlgorithm(t, t/2, 10, nnop);
            train(oa, network, fileName, writer); //trainer.train();

            double trainAcc = testAccuracy(trainInstances, oa, network);
            double testAcc = testAccuracy(testInstances, oa, network);

            writer.close();
            t += 50;
        }
    }

    private static void testSA1(){
        double t = 0.95;
        for (int i = 0; i < 3; i++) {
            BackPropagationNetwork network = factory.createClassificationNetwork(
                    new int[]{inputLayer, hiddenLayer, outputLayer});
            NeuralNetworkOptimizationProblem nnop = new NeuralNetworkOptimizationProblem(trainSet, network, measure);
            String fileName = "SA1_test" + i + ".csv";
            MyWriter writer = new MyWriter(fileName, new String[]{"Iterations", "Error", "Train accuracy", "Test Accuracy", "time"});
            OptimizationAlgorithm oa = new SimulatedAnnealing(100, t, nnop);
            train(oa, network, fileName, writer); //trainer.train();

            writer.close();
            t -= 0.15;
        }
    }

    private static void testSA0(){
        double t = 0.01;
        for (int i = 0; i < 6; i++) {
            BackPropagationNetwork network = factory.createClassificationNetwork(
                    new int[]{inputLayer, hiddenLayer, outputLayer});
            NeuralNetworkOptimizationProblem nnop = new NeuralNetworkOptimizationProblem(trainSet, network, measure);
            String fileName = "SA0_test" + i + ".csv";
            MyWriter writer = new MyWriter(fileName, new String[]{"Iterations", "Error", "Train accuracy", "Test Accuracy", "time"});
            OptimizationAlgorithm oa = new SimulatedAnnealing(t, 0.95, nnop);
            train(oa, network, fileName, writer); //trainer.train();

            double trainAcc = testAccuracy(trainInstances, oa, network);
            double testAcc = testAccuracy(testInstances, oa, network);

            writer.close();
            t = t * 100;
        }
    }

    private static void myTest(){
        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                    new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(trainSet, networks[i], measure);
        }

        errorWriters[0] = RHCErrorWriter;
        errorWriters[1] = SAErrorWriter;
        errorWriters[2] = GAErrorWriter;

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(1e11, .80, nnop[1]);
        oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

        for(int i = 0; i < 1; i++) {
            double start = System.nanoTime(), end, trainingTime, testingTime;
            train(oa[i], networks[i], oaNames[i], errorWriters[i]); //trainer.train();
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);

            double trainAcc = testAccuracy(trainInstances, oa[i], networks[i]);
            double testAcc = testAccuracy(testInstances, oa[i], networks[i]);


            results +=  "\nResults for " + oaNames[i] + ": \ntraining accuracy: " + trainAcc +
                    "\ntest accuracy " + testAcc + "%\nTraining time: " + df.format(trainingTime)
                    + " seconds\n";
        }

        System.out.println(results);
    }

    private static double testAccuracy(Instance[] instances, OptimizationAlgorithm oa, BackPropagationNetwork network){


        Instance optimalInstance = oa.getOptimal();
        network.setWeights(optimalInstance.getData());

        double predicted, actual;
        double correct = 0, incorrect = 0;

        for(int j = 0; j < instances.length; j++) {
            network.setInputValues(instances[j].getData());
            network.run();

            predicted = Double.parseDouble(instances[j].getLabel().toString());
            actual = Double.parseDouble(network.getOutputValues().toString());
            //System.out.println(predicted + ", "+ actual);
            double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

//                actual = getLabel(instances[j].getLabel().getData());
//
//                Vector res = network.getOutputValues();
//                predicted = getLabel(res);
//                //Double.parseDouble(networks[i].getOutputValues().toString());
//
//                if (actual == predicted) correct++;
//                else incorrect++;

        }
        return correct/(correct+incorrect)*100;
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName, MyWriter errorWriter) {
        System.out.println("\nError results for " + oaName + "\n---------------------------");
        double testAcc = 0;
        double trainAcc = 0;
        long startTime = System.nanoTime();

        int iterations = trainingIterations;
        if (oaName.equals("GA"))
            iterations = iterations / 2;
        errorWriter.open();

        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            if (i == 0 || (i + 1) % (testInterval) == 0) {
                double time = (double) (System.nanoTime() - startTime) / 1e9;
                double error = 0;
                for (int j = 0; j < trainInstances.length; j++) {
                    network.setInputValues(trainInstances[j].getData());
                    network.run();

                    Instance output = trainInstances[j].getLabel(), example = new Instance(network.getOutputValues());
//                    int lb = getLabel(network.getOutputValues());
//                    example.setLabel(getOutputInstance(lb));
                    example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                    error += measure.value(output, example);
                }
                errorWriter.write(Integer.toString(i + 1));
                errorWriter.write(df.format(error).toString());

                trainAcc = testAccuracy(trainInstances, oa, network);
                testAcc = testAccuracy(testInstances, oa, network);

                errorWriter.write(df.format(trainAcc));
                errorWriter.write(df.format(testAcc));

                errorWriter.write(df.format(time));
                errorWriter.finishRow();

                System.out.println(i + " " + df.format(error).toString() + " " + df.format(trainAcc) + " " + df.format(testAcc));
            }
        }
        errorWriter.close();
    }

    private static Instance[] readInstances(String pathName){
        int tot = 0;
        int attributeNum = -1;
        try {
            LineNumberReader lnr = new LineNumberReader(new FileReader(new File(pathName)));
            lnr.skip(Long.MAX_VALUE);
            tot = lnr.getLineNumber();
            System.out.println(tot);
        }
        catch(Exception e){
            e.printStackTrace();
        }

        try{
            BufferedReader br = new BufferedReader(new FileReader(new File(pathName)));
            Scanner scan = new Scanner((br.readLine()));
            scan.useDelimiter(",");
            while (scan.hasNext()) {
                attributeNum++;
                scan.next();
            }
            System.out.println(attributeNum);
        }
        catch (Exception e){
            e.printStackTrace();
        }

        return initializeInstances(pathName, tot, attributeNum);

    }

    private static Instance[] initializeInstances(String pathName, int dataNum, int attributeNum) {

        double[][][] attributes = new double[dataNum][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File(pathName)));
            //System.out.println(attributes.length);
            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[attributeNum]; // 7 attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < attributeNum; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                String st = scan.next().toString();
                //System.out.println(st);
                if (st.equals("g")) attributes[i][1][0] = 0;
                else attributes[i][1][0] = 1;
                //System.out.println(attributes[i][1][0]);
                //attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            // classifications range from 0 to 30; split into 0 - 14 and 15 - 30
            instances[i].setLabel(new Instance(attributes[i][1][0]));
            //instances[i].setLabel(getOutputInstance((int) attributes[i][1][0]));

//            if (attributes[i][1][0] < 15)
//                instances[i].setLabel(getOutputInstance(0));
//            else instances[i].setLabel(getOutputInstance(1));
            //System.out.println(instances[i].getLabel());
        }

        return instances;
    }

}
