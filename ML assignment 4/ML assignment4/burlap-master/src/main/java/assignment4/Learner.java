package assignment4;

import burlap.behavior.policy.GreedyQPolicy;
import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.learning.tdmethods.SarsaLam;
import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.valuefunction.QProvider;
import burlap.behavior.valuefunction.ValueFunction;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldRewardFunction;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridLocation;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.auxiliary.stateconditiontest.StateConditionTest;
import burlap.mdp.auxiliary.stateconditiontest.TFGoalCondition;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.oo.state.generic.GenericOOState;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.shell.visual.VisualExplorer;
import burlap.statehashing.HashableStateFactory;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.Visualizer;

import java.util.List;
import java.util.Random;

/**
 * Created by allam on 4/18/17.
 */
public class Learner {
    public GridWorldDomain gridWorld;
    OOSADomain domain;
    GridWorldTerminalFunction tf;
    GridWorldRewardFunction rf;
    StateConditionTest goalCondition;
    State initialState;
    HashableStateFactory hashingFactory;
    SimulatedEnvironment env;

    public void InitTest1(){
        int width, height;
        int[][] map = new int[][]{
                {0,0,0,1,0,0},
                {0,0,0,0,0,0},
                {0,0,0,1,0,0},
                {0,0,1,1,1,0},
                {0,0,0,0,1,0},
                {1,0,0,0,0,0}
        };
        width = map.length;
        height = map[0].length;
        gridWorld = new GridWorldDomain(map);
        gridWorld.setProbSucceedOrthogonal(0.8);


        rf = new GridWorldRewardFunction(width, height, -1.0);
        rf.setReward(width-1, height-1, 100);
        rf.setReward(width-1, height-2, -100);
        tf = new GridWorldTerminalFunction(width-1, height-1);
        tf.markAsTerminalPosition(width-1,height-2);

        gridWorld.setTf(tf);
        gridWorld.setRf(rf);
        goalCondition = new TFGoalCondition(tf);

        domain = gridWorld.generateDomain();

        initialState = new GridWorldState(new GridAgent(0, 0),
                new GridLocation(width-1, width-1, "loc0"),
                new GridLocation(width-1, height-2, "loc1"));

        hashingFactory = new SimpleHashableStateFactory();
        env = new SimulatedEnvironment(domain, initialState);
    }

    void SetSquare(int x, int y, int size, int v, int[][] map){
        for (int i = x; i < x+size; i++)
            for (int j = y; j < y+size; j++)
                map[i][j] = v;
    }

    void SetCrossing(int x, int y, int t, int[][] map){
        SetCrossing(x-t, y-t, 2,1,  map);
        SetCrossing(x-t, y+t, 2, 1, map);
        SetCrossing(x+t, y+t, 2, 1, map);
        SetCrossing(x+t, y-t, 2, 1, map);
        SetCrossing(x, y, 2, 1, map);
    }

    void SetCrossing(int x, int y, int size, int v, int[][] map){
        for (int i = x-size; i <= x+size; i++)
            map[i][y] = v;

        for (int j = y-size; j <= y+size; j++)
            map[x][j] = v;
    }

    int[][] GetMDP2(int width, int height){
        int mid = width/2;
        int[][] map = new int[width][height];
        for (int i = 0; i < width; i++){
            for (int j = 0; j < height; j++){
                map[i][j] = 0;
            }
        }

        for (int i = 0; i < width; i++){
            map[i][mid] = 1; map[i][mid-1] = 1;
            map[mid][i] = 1; map[mid-1][i] = 1;
        }

        SetSquare(mid-1, mid/2-1, 2, 0, map);
        SetSquare(mid/2-1, mid-1, 2, 0, map);
        SetSquare((width+mid)/2-1, mid-1, 2, 0, map);
        SetSquare(mid-1, (width+mid)/2-1, 2, 0, map);

        int t = 5;
        SetCrossing(mid/2-1, mid/2-1, t,  map);
        SetCrossing(mid/2-1, (width+mid)/2-1, t,  map);
        SetCrossing((width+mid)/2-1, mid/2-1, t, map);
        SetCrossing((width+mid)/2-1, (width+mid)/2-1, t, map);
        return map;
    }
    public void InitTest2(){
        int width = 40;
        int height = 40;
        int[][] map = GetMDP2(width, height);

        gridWorld = new GridWorldDomain(map);
        gridWorld.setProbSucceedOrthogonal(0.95);
        rf = new GridWorldRewardFunction(width, height, -1.0);
        tf = new GridWorldTerminalFunction(width-1, height-1);

        int randomNum = 200;
        Random random = new Random(15);
        for (int i = 0; i < randomNum; i++){
            int x = (random.nextInt() % width + width) % width;
            int y = (random.nextInt() % height + height) % height;
            //System.out.println(x + ", " + y);
            if (map[x][y] == 0)
                rf.setReward(x, y, -20);
        }


        gridWorld.setTf(tf);
        gridWorld.setRf(rf);
        goalCondition = new TFGoalCondition(tf);

        domain = gridWorld.generateDomain();

        initialState = new GridWorldState(new GridAgent(0, 0),
                new GridLocation(width-1, width-1, "loc0"));

        hashingFactory = new SimpleHashableStateFactory();
        env = new SimulatedEnvironment(domain, initialState);
    }

    public void InitGridWorld(int size){
        int width = size;
        int height = size;

        gridWorld = new GridWorldDomain(width, height);
        rf = new GridWorldRewardFunction(width, height, -1.0);
        tf = new GridWorldTerminalFunction(width-1, height-1);


        gridWorld.setTf(tf);
        gridWorld.setRf(rf);
        goalCondition = new TFGoalCondition(tf);

        domain = gridWorld.generateDomain();

        initialState = new GridWorldState(new GridAgent(0, 0),
                new GridLocation(width-1, width-1, "loc0"));

        hashingFactory = new SimpleHashableStateFactory();
        env = new SimulatedEnvironment(domain, initialState);
    }

    public Learner(int test){
        if (test == 2)
            InitTest2();
        else InitTest1();
    }

    public Learner(){

    }


    public void CompareGridWorld(){
        AnalysisData data1 = new AnalysisData();
        AnalysisData data2 = new AnalysisData();

        for (int i = 4; i <= 40; i++){
            InitGridWorld(i);
            DoValueIteration("", data1);
            DoPolicyIteration("", data2);
        }

        data1.WriteFile("python/exp_compare_vi");
        data2.WriteFile("python/exp_compare_pi");
    }

    public void visualize(String outputPath){
        Visualizer v = GridWorldVisualizer.getVisualizer(gridWorld.getMap());
        new EpisodeSequenceVisualizer(v, domain, outputPath);
    }

    public void ValueFunctionVisulization(ValueFunction valueFunction, Policy p){
        List<State> allStates = StateReachability.getReachableStates(initialState, domain, hashingFactory);

        ValueFunctionVisualizerGUI gui = GridWorldDomain.getGridWorldValueFunctionVisualization(
                allStates, gridWorld.getWidth(), gridWorld.getHeight(), valueFunction, p);

        gui.initGUI();
    }

    public void DoValueIteration(String outputPath, AnalysisData data){
        int maxk = 10;
        double time = 0;

        Planner planner=null;
        Policy p = null;

        for (int k = 0; k < maxk; k++){
            long startTime = System.nanoTime();
            planner = new ValueIteration(domain, 0.99,
                    hashingFactory, 0.001, 1);
            p = planner.planFromState(initialState);
            time += (System.nanoTime() - startTime) / 1e6;
        }

        time /= maxk;

        if (data != null){
            data.AddData(gridWorld.getHeight(), 0,
                    ((ValueIteration)planner).totalIterationNumber, time);
        }

        System.out.println("Value Iteration data:");
        System.out.println("average time: " + time);
        Episode ep = PolicyUtils.rollout(p, initialState, domain.getModel());
        System.out.println(ep.maxTimeStep() + ", " + ep.discountedReturn(1.0));
        System.out.println("\n");

        if (!outputPath.equals("")) {
            ep.write(outputPath + "vi");
            ValueFunctionVisulization((ValueFunction) planner, p);
        }
    }

    public void DoPolicyIteration(String outputPath, AnalysisData data){
        int maxk = 10;
        double time = 0;

        Planner planner=null;
        Policy p = null;

        for (int k = 0; k < maxk; k++){
            long startTime = System.nanoTime();
            planner = new PolicyIteration(domain, 0.99,
                    hashingFactory, 0.001, 100, 1);
            p = planner.planFromState(initialState);
            time += (System.nanoTime() - startTime) / 1e6;
        }

        time /= maxk;

        if (data != null){
            data.AddData(gridWorld.getHeight(), 0,
                    ((PolicyIteration)planner).getTotalPolicyIterations(), time);
        }

        System.out.println("Policy Iteration data:");
        System.out.println("average time: " + time);
        Episode ep = PolicyUtils.rollout(p, initialState, domain.getModel());
        System.out.println(ep.maxTimeStep() + ", " + ep.discountedReturn(1.0));
        System.out.println("\n");

        if (!outputPath.equals("")) {
            ep.write(outputPath + "vi");
            ValueFunctionVisulization((ValueFunction) planner, p);
        }
    }

    public void DoQLearning(String outputPath, String expName, int iterations){
        AnalysisData data = new AnalysisData();
        int maxk = 10;
        for (int k = 0; k < maxk; k++) {
            LearningAgent agent = new QLearning(domain, 0.99, hashingFactory, 0, 0.8);

            for (int i = 0; i < iterations; i++) {
                Episode e = agent.runLearningEpisode(env);

                System.out.println(i + ": " + e.maxTimeStep() + ", " + e.discountedReturn(1.0));


                if (i == iterations - 1)
                    e.write(outputPath + "QLearning");
                env.resetEnvironment();

                if (k == 0)
                    data.AddData(i, e.discountedReturn(1.0), e.maxTimeStep(), 0);
                else {
                    double t = data.expectedValue.get(i);
                    t += e.discountedReturn(1.0);
                    if (k == maxk-1)
                        t = t / maxk;
                    data.expectedValue.set(i, t);
                }
            }

            if (k == maxk-1) {
                ValueFunctionVisulization((ValueFunction) agent, new GreedyQPolicy((QProvider) agent));
                data.WriteFile("python/" + expName + "_qlearning.csv");
            }
        }
    }

    public void DoSarsaLearning(String outputPath){
        LearningAgent agent = new SarsaLam(domain, 0.99, hashingFactory, 0, 0.5, 0.3);
        int iterations = 50;

        for (int i = 0; i < iterations; i++){
            Episode e = agent.runLearningEpisode(env);

            System.out.println(i + ": " + e.maxTimeStep());
            if (i == iterations-1)
                e.write(outputPath + "Sarsa");
            env.resetEnvironment();
        }
        ValueFunctionVisulization((ValueFunction)agent, new GreedyQPolicy((QProvider)agent));
    }

    public static void MDP1Test(){
        Learner learner = new Learner(1);
        String outputPath = "output/";

        learner.CompareGridWorld();

        learner.DoValueIteration(outputPath, null);
        learner.DoPolicyIteration(outputPath,null);
        learner.DoQLearning(outputPath, "MDP1", 200);

        learner.visualize(outputPath);
    }

    public static void MDP2Test(){
        Learner learner = new Learner(2);
        String outputPath = "output/";


        learner.DoValueIteration(outputPath, null);
        learner.DoPolicyIteration(outputPath,null);
        learner.DoQLearning(outputPath, "MDP2", 1000);

        learner.visualize(outputPath);
    }

    public static void main(String[] args){
        Learner.MDP1Test();

        Learner.MDP2Test();
    }

}
