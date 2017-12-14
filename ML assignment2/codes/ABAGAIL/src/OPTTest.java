/**
 * Created by allam on 3/9/17.
 */

import opt.test.*;

public class OPTTest {

    private static void testFlipFlop(){
        FlipFlopTest flipFlopTest = new FlipFlopTest();
        //flipFlopTest.SATest_T();
        flipFlopTest.RHCTest();
        flipFlopTest.SATest();
        flipFlopTest.GATest();
        flipFlopTest.MIMICTest();
    }

    private static void testCountOnes(){
        CountOnesTest countOnesTest = new CountOnesTest();
        countOnesTest.RHCTest();
        countOnesTest.SATest();
        countOnesTest.GATest();
        countOnesTest.MIMICTest();
    }

    private static void testTSP(){
        TravelingSalesmanTest travelingSalesmanTest = new TravelingSalesmanTest();
        travelingSalesmanTest.RHCTest();
        travelingSalesmanTest.SATest();
        travelingSalesmanTest.GATest();
        travelingSalesmanTest.MIMICTest();
    }

    private static void testKnapsack(){
        KnapsackTest knapsackTest = new KnapsackTest();
        knapsackTest.RHCTest();
        knapsackTest.SATest();
        knapsackTest.GATest();
        knapsackTest.MIMICTest();
    }

    public static void main(String[] args){
        testFlipFlop();
        testTSP();
        testKnapsack();
    }
}
