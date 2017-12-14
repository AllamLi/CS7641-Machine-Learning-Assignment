package opt.test;

import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.*;
import opt.example.*;
import opt.ga.*;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;
import shared.writer.*;

/**
 * 
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class TravelingSalesmanTest {
    /** The n value */
    private static final int N = 50;
    /**
     * The test main
     * @param args ignored
     */

    private int RHCTestTimes = 20;
    private int SATestTimes = 20;
    private int GATestTimes = 20;
    private int MIMICTestTimes = 5;

    private int RHCBg = 40, RHCMax = 5000;
    private int SABg = 40, SAMax = 5000;
    private int GABg = 10, GAMax = 1000;
    private int MIMICBg = 40, MIMICMax = 1000;

    double[][] points = new double[N][2];

    public TravelingSalesmanTest(){
        Random random = new Random();
        for (int i = 0; i < points.length; i++) {
            points[i][0] = random.nextDouble();
            points[i][1] = random.nextDouble();
        }
    }



    public void RHCTest(){
        MyWriter writer = new MyWriter("TSP_RHC.csv", new String[]{"iterations", "fitness value", "time"});
        writer.open();
        TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
        Distribution odd = new DiscretePermutationDistribution(N);
        NeighborFunction nf = new SwapNeighbor();
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);


        System.out.println("\n");
        System.out.println("RHC of TSP Problem:\n");
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
        TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
        Distribution odd = new DiscretePermutationDistribution(N);
        NeighborFunction nf = new SwapNeighbor();
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);

        System.out.println("\n");
        System.out.println("SA of TSP Problem:\n");

        MyWriter writer = new MyWriter("TSP_SA.csv", new String[]{"iterations", "fitness value", "time"});
        writer.open();
        for (int it = SABg; it <= SAMax; it+=SABg) {
            double sum = 0;
            long totTime = 0;
            for (int i = 0; i < SATestTimes; i++) {
                SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .95, hcp);
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
        TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
        Distribution odd = new DiscretePermutationDistribution(N);
        MutationFunction mf = new SwapMutation();
        CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);

        System.out.println("\n");
        System.out.println("GA of TSP Problem:\n");

        MyWriter writer = new MyWriter("TSP_GA.csv", new String[]{"iterations", "fitness value", "time"});
        writer.open();
        for (int it = GABg; it <= RHCMax; it = it < GAMax ? (it+GABg): (it+(RHCMax-GAMax)/10)) {
            double sum = 0;
            long totTime = 0;
            for (int i = 0; i < GATestTimes; i++) {
                StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 20, gap);
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
        TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanSortEvaluationFunction(points);
        int[] ranges = new int[N];
        Arrays.fill(ranges, N);
        Distribution odd = new  DiscreteUniformDistribution(ranges);
        Distribution df = new DiscreteDependencyTree(.1, ranges);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        System.out.println("\n");
        System.out.println("MIMIC of TSP Problem:\n");

        MyWriter writer = new MyWriter("TSP_MIMIC.csv", new String[]{"iterations", "fitness value", "time"});
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

    public static void main(String[] args) {
        Random random = new Random();
        // create the random points
        double[][] points = new double[N][2];
        for (int i = 0; i < points.length; i++) {
            points[i][0] = random.nextDouble();
            points[i][1] = random.nextDouble();   
        }
        // for rhc, sa, and ga we use a permutation based encoding
        TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
        Distribution odd = new DiscretePermutationDistribution(N);
        NeighborFunction nf = new SwapNeighbor();
        MutationFunction mf = new SwapMutation();
        CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200000);
        fit.train();
        System.out.println(ef.value(rhc.getOptimal()));
        
        SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .95, hcp);
        fit = new FixedIterationTrainer(sa, 200000);
        fit.train();
        System.out.println(ef.value(sa.getOptimal()));
        
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 20, gap);
        fit = new FixedIterationTrainer(ga, 1000);
        fit.train();
        System.out.println(ef.value(ga.getOptimal()));
        
        // for mimic we use a sort encoding
        ef = new TravelingSalesmanSortEvaluationFunction(points);
        int[] ranges = new int[N];
        Arrays.fill(ranges, N);
        odd = new  DiscreteUniformDistribution(ranges);
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        
        MIMIC mimic = new MIMIC(200, 100, pop);
        fit = new FixedIterationTrainer(mimic, 1000);
        fit.train();
        System.out.println(ef.value(mimic.getOptimal()));
        
    }
}
