package opt.test;

import java.util.Arrays;

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
import opt.ga.*;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * 
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class CountOnesTest {
    /** The n value */
    private static final int N = 100;

    private int RHCBg = 50, RHCMax = 1000;
    private int SABg = 50, SAMax = 1000;
    private int GABg = 50, GAMax = 1000;
    private int MIMICBg = 10, MIMICMax = 100;

    private int RHCTestTimes = 100;
    private int SATestTimes = 100;
    private int GATestTimes = 100;
    private int MIMICTestTimes = 10;

    int[] ranges = new int[N];


    public CountOnesTest(){
        Arrays.fill(ranges, 2);
    }

    public void RHCTest(){

        EvaluationFunction ef = new CountOnesEvaluationFunction();
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);


        System.out.println("\n");
        System.out.println("RHC of Count Ones Problem:\n");
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
        }
    }

    public void SATest(){
        EvaluationFunction ef = new FlipFlopEvaluationFunction();

        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);

        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);

        System.out.println("\n");
        System.out.println("SA of Count Ones Problem:\n");
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
        }
    }

    public void GATest(){
        EvaluationFunction ef = new CountOnesEvaluationFunction();
        Distribution odd = new DiscreteUniformDistribution(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new UniformCrossOver();
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);

        System.out.println("\n");
        System.out.println("GA of Count Ones Problem:\n");
        for (int it = GABg; it <= GAMax; it += GABg) {
            double sum = 0;
            long totTime = 0;
            for (int i = 0; i < GATestTimes; i++) {
                StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(20, 20, 0, gap);
                long bgTime = System.nanoTime();
                FixedIterationTrainer fit = new FixedIterationTrainer(ga, it);
                fit.train();
                sum += ef.value(ga.getOptimal());
                totTime += System.nanoTime() - bgTime;
            }
            sum /= GATestTimes;
            double outputTime = (double) (totTime) / (GATestTimes * 1e9d);
            System.out.println(it + ", " + sum + ", " + outputTime);
        }
    }

    public void MIMICTest(){
        EvaluationFunction ef = new CountOnesEvaluationFunction();
        Distribution odd = new DiscreteUniformDistribution(ranges);
        Distribution df = new DiscreteDependencyTree(.1, ranges);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        System.out.println("\n");
        System.out.println("MIMIC of Count Ones Problem:\n");
        for (int it = MIMICBg; it <= MIMICMax; it += MIMICBg) {
            double sum = 0;
            long totTime = 0;
            for (int i = 0; i < MIMICTestTimes; i++) {
                MIMIC mimic = new MIMIC(50, 10, pop);
                long bgTime = System.nanoTime();
                FixedIterationTrainer fit = new FixedIterationTrainer(mimic, it);
                fit.train();
                sum += ef.value(mimic.getOptimal());
                totTime += System.nanoTime() - bgTime;
            }
            sum /= MIMICTestTimes;
            double outputTime = (double) (totTime) / (MIMICTestTimes * 1e9d);
            System.out.println(it + ", " + sum + ", " + outputTime);
        }
    }
    
    public static void main(String[] args) {
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new CountOnesEvaluationFunction();
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new UniformCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200);
        fit.train();
        System.out.println(ef.value(rhc.getOptimal()));
        
        SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
        fit = new FixedIterationTrainer(sa, 200);
        fit.train();
        System.out.println(ef.value(sa.getOptimal()));
        
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(20, 20, 0, gap);
        fit = new FixedIterationTrainer(ga, 300);
        fit.train();
        System.out.println(ef.value(ga.getOptimal()));
        
        MIMIC mimic = new MIMIC(50, 10, pop);
        fit = new FixedIterationTrainer(mimic, 100);
        fit.train();
        System.out.println(ef.value(mimic.getOptimal()));
    }
}