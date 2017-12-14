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
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;
import shared.writer.*;
/**
 * A test using the flip flop evaluation function
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class FlipFlopTest {
    /** The n value */
    private static final int N = 200;

    private int RHCTestTimes = 20;
    private int SATestTimes = 20;
    private int GATestTimes = 20;
    private int MIMICTestTimes = 5;

    private int RHCBg = 40, RHCMax = 5000;
    private int SABg = 40, SAMax = 5000;
    private int GABg = 40, GAMax = 4000;
    private int MIMICBg = 5, MIMICMax = 200;

    int[] ranges = new int[N];

    public FlipFlopTest(){
        Arrays.fill(ranges, 2);
    }

    public void RHCTest(){
        EvaluationFunction ef = new FlipFlopEvaluationFunction();


        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);

        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);

        System.out.println("\n");
        System.out.println("RHC of Flip Flop Problem:\n");

        MyWriter writer = new MyWriter("FF_RHC.csv", new String[]{"iterations", "fitness value", "time"});
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

    public void SATest_T(){
        EvaluationFunction ef = new FlipFlopEvaluationFunction();

        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);

        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);

        System.out.println("\n");
        System.out.println("SA of Flip Flop Problem:\n");

        double T = .95;
        for (int number = 0; number < 3; number++) {
            MyWriter writer = new MyWriter("FF_SA" + number + ".csv", new String[]{"iterations", "fitness value", "time"});
            writer.open();
            for (int it = SABg; it <= SAMax; it += SABg) {
                double sum = 0;
                long totTime = 0;
                for (int i = 0; i < SATestTimes; i++) {
                    SimulatedAnnealing sa = new SimulatedAnnealing(100, T, hcp);
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

            T -= 0.2;
            writer.close();
        }

    }

    public void SATest(){
        EvaluationFunction ef = new FlipFlopEvaluationFunction();

        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);

        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);

        System.out.println("\n");
        System.out.println("SA of Flip Flop Problem:\n");
        MyWriter writer = new MyWriter("FF_SA.csv", new String[]{"iterations", "fitness value", "time"});
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
        EvaluationFunction ef = new FlipFlopEvaluationFunction();

        Distribution odd = new DiscreteUniformDistribution(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);

        CrossoverFunction cf = new SingleCrossOver();

        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);

        System.out.println("\n");
        System.out.println("GA of Flip Flop Problem:\n");
        MyWriter writer = new MyWriter("FF_GA.csv", new String[]{"iterations", "fitness value", "time"});
        writer.open();
        for (int it = GABg; it <= RHCMax; it = it < GAMax ? (it+GABg): (it+(RHCMax-GAMax)/10)) {
            double sum = 0;
            long totTime = 0;
            for (int i = 0; i < GATestTimes; i++) {
                StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 20, gap);
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
        EvaluationFunction ef = new FlipFlopEvaluationFunction();

        Distribution odd = new DiscreteUniformDistribution(ranges);

        Distribution df = new DiscreteDependencyTree(.1, ranges);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        System.out.println("\n");
        System.out.println("MIMIC of Flip Flop Problem:\n");
        MyWriter writer = new MyWriter("FF_MIMIC.csv", new String[]{"iterations", "fitness value", "time"});
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
        int[] ranges = new int[N];
        EvaluationFunction ef = new FlipFlopEvaluationFunction();

        Arrays.fill(ranges, 2);

        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
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
        
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 20, gap);
        fit = new FixedIterationTrainer(ga, 1000);
        fit.train();
        System.out.println(ef.value(ga.getOptimal()));
        
        MIMIC mimic = new MIMIC(200, 5, pop);
        fit = new FixedIterationTrainer(mimic, 1000);
        fit.train();
        System.out.println(ef.value(mimic.getOptimal()));
    }
}
