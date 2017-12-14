package opt.test;

import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;
import shared.writer.MyWriter;

/**
 * A test of the knapsack problem
 *
 * Given a set of items, each with a weight and a value, determine the number of each item to include in a
 * collection so that the total weight is less than or equal to a given limit and the total value is as
 * large as possible.
 * https://en.wikipedia.org/wiki/Knapsack_problem
 *
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class KnapsackTest {
    /** Random number generator */
    private static final Random random = new Random();
    /** The number of items */
    private static final int NUM_ITEMS = 40;
    /** The number of copies each */
    private static final int COPIES_EACH = 4;
    /** The maximum value for a single element */
    private static final double MAX_VALUE = 50;
    /** The maximum weight for a single element */
    private static final double MAX_WEIGHT = 50;
    /** The maximum weight for the knapsack */
    private static final double MAX_KNAPSACK_WEIGHT =
         MAX_WEIGHT * NUM_ITEMS * COPIES_EACH * .4;

    int[] copies = new int[NUM_ITEMS];
    double[] values = new double[NUM_ITEMS];
    double[] weights = new double[NUM_ITEMS];
    int[] ranges = new int[NUM_ITEMS];

    public KnapsackTest(){
        for (int i = 0; i < NUM_ITEMS; i++) {
            values[i] = random.nextDouble() * MAX_VALUE;
            weights[i] = random.nextDouble() * MAX_WEIGHT;
        }
        Arrays.fill(ranges, COPIES_EACH + 1);
    }


    private int RHCTestTimes = 20;
    private int SATestTimes = 20;
    private int GATestTimes = 20;
    private int MIMICTestTimes = 5;

    private int RHCBg = 40, RHCMax = 5000;
    private int SABg = 40, SAMax = 5000;
    private int GABg = 10, GAMax = 1000;
    private int MIMICBg = 5, MIMICMax = 200;

    public void RHCTest(){
        EvaluationFunction ef = new KnapsackEvaluationFunction(values, weights, MAX_KNAPSACK_WEIGHT, copies);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);

        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);

        System.out.println("\n");
        System.out.println("RHC of Knapsack Problem:\n");

        MyWriter writer = new MyWriter("KNS_RHC.csv", new String[]{"iterations", "fitness value", "time"});
        writer.open();
        for (int it = RHCBg; it <= RHCMax; it+=RHCBg) {
            double sum = 0;
            long totTime = 0;
            for (int i = 0; i < RHCTestTimes; i++) {
                RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
                long bgTime = System.nanoTime();
                FixedIterationTrainer fit = new FixedIterationTrainer(rhc, it);
                fit.train();
                sum += ef.value(rhc.getOptimal());
                totTime += System.nanoTime() - bgTime;
            }
            sum /= RHCTestTimes;
            double outputTime = (double) (totTime) / (RHCTestTimes * 1e9d);
            System.out.println(it + ", " + sum + ", " + outputTime);
            writer.write(Integer.toString(it));
            writer.write(Double.toString(sum));
            writer.write(Double.toString(outputTime));
            writer.finishRow();
        }
        writer.close();
    }

    public void SATest(){
        EvaluationFunction ef = new KnapsackEvaluationFunction(values, weights, MAX_KNAPSACK_WEIGHT, copies);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);

        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);

        System.out.println("\n");
        System.out.println("SA of Knapsack Problem:\n");
        MyWriter writer = new MyWriter("KNS_SA.csv", new String[]{"iterations", "fitness value", "time"});
        writer.open();
        for (int it = SABg; it <= SAMax; it+=SABg) {
            double sum = 0;
            long totTime = 0;
            for (int i = 0; i < SATestTimes; i++) {
                SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
                long bgTime = System.nanoTime();
                FixedIterationTrainer fit = new FixedIterationTrainer(sa, it);
                fit.train();
                sum += ef.value(sa.getOptimal());
                totTime += System.nanoTime() - bgTime;
            }
            sum /= SATestTimes;
            double outputTime = (double) (totTime) / (SATestTimes * 1e9d);
            System.out.println(it + ", " + sum + ", " + outputTime);
            writer.write(Integer.toString(it));
            writer.write(Double.toString(sum));
            writer.write(Double.toString(outputTime));
            writer.finishRow();
        }
        writer.close();
    }

    public void GATest(){
        EvaluationFunction ef = new KnapsackEvaluationFunction(values, weights, MAX_KNAPSACK_WEIGHT, copies);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new UniformCrossOver();
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);

        System.out.println("\n");
        System.out.println("GA of Knapsack Problem:\n");

        MyWriter writer = new MyWriter("KNS_GA.csv", new String[]{"iterations", "fitness value", "time"});
        writer.open();

        for (int it = GABg; it <= RHCMax; it = it < GAMax ? (it+GABg): (it+(RHCMax-GAMax)/10)) {
            double sum = 0;
            long totTime = 0;
            for (int i = 0; i < GATestTimes; i++) {
                StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 25, gap);
                long bgTime = System.nanoTime();
                FixedIterationTrainer fit = new FixedIterationTrainer(ga, it);
                fit.train();
                sum += ef.value(ga.getOptimal());
                totTime += System.nanoTime() - bgTime;
            }
            sum /= GATestTimes;
            double outputTime = (double) (totTime) / (GATestTimes * 1e9d);
            System.out.println(it + ", " + sum + ", " + outputTime);
            writer.write(Integer.toString(it));
            writer.write(Double.toString(sum));
            writer.write(Double.toString(outputTime));
            writer.finishRow();
        }
        writer.close();
    }

    public void MIMICTest(){
        EvaluationFunction ef = new KnapsackEvaluationFunction(values, weights, MAX_KNAPSACK_WEIGHT, copies);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        Distribution df = new DiscreteDependencyTree(.1, ranges);

        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        System.out.println("\n");
        System.out.println("MIMIC of Knapsack Problem:\n");

        MyWriter writer = new MyWriter("KNS_MIMIC.csv", new String[]{"iterations", "fitness value", "time"});
        writer.open();

        for (int it = MIMICBg; it <= RHCMax; it = it < MIMICMax ? (it+MIMICBg): (it+(RHCMax-MIMICMax)/10)) {
            double sum = 0;
            long totTime = 0;
            for (int i = 0; i < MIMICTestTimes; i++) {
                MIMIC mimic = new MIMIC(200, 100, pop);
                long bgTime = System.nanoTime();
                FixedIterationTrainer fit = new FixedIterationTrainer(mimic, it);
                fit.train();
                sum += ef.value(mimic.getOptimal());
                totTime += System.nanoTime() - bgTime;
            }
            sum /= MIMICTestTimes;
            double outputTime = (double) (totTime) / (MIMICTestTimes * 1e9d);
            System.out.println(it + ", " + sum + ", " + outputTime);
            writer.write(Integer.toString(it));
            writer.write(Double.toString(sum));
            writer.write(Double.toString(outputTime));
            writer.finishRow();
        }

        writer.close();
    }

    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
        int[] copies = new int[NUM_ITEMS];
        Arrays.fill(copies, COPIES_EACH);
        double[] values = new double[NUM_ITEMS];
        double[] weights = new double[NUM_ITEMS];
        for (int i = 0; i < NUM_ITEMS; i++) {
            values[i] = random.nextDouble() * MAX_VALUE;
            weights[i] = random.nextDouble() * MAX_WEIGHT;
        }
        int[] ranges = new int[NUM_ITEMS];
        Arrays.fill(ranges, COPIES_EACH + 1);

        EvaluationFunction ef = new KnapsackEvaluationFunction(values, weights, MAX_KNAPSACK_WEIGHT, copies);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);

        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new UniformCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges);

        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200000);
        fit.train();
        System.out.println(ef.value(rhc.getOptimal()));
        
        SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
        fit = new FixedIterationTrainer(sa, 200000);
        fit.train();
        System.out.println(ef.value(sa.getOptimal()));
        
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 25, gap);
        fit = new FixedIterationTrainer(ga, 1000);
        fit.train();
        System.out.println(ef.value(ga.getOptimal()));
        
        MIMIC mimic = new MIMIC(200, 100, pop);
        fit = new FixedIterationTrainer(mimic, 1000);
        fit.train();
        System.out.println(ef.value(mimic.getOptimal()));
    }

}
