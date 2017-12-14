package assignment4;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by allam on 4/20/17.
 */
public class AnalysisData {
    public List<Integer> iterationNumber = new ArrayList<Integer>();

    public List<Double> expectedValue = new ArrayList<Double>();
    public List<Integer> maxSteps = new ArrayList<Integer>();
    public List<Double> usedTime = new ArrayList<Double>();

    private static final String COMMA_DELIMITER = ",";

    private static final String NEW_LINE_SEPARATOR = "\n";


    public AnalysisData(){

    }

    public void Reset(){
        iterationNumber = new ArrayList<Integer>();

        expectedValue = new ArrayList<Double>();
        maxSteps = new ArrayList<Integer>();
        usedTime = new ArrayList<Double>();
    }

    public void AddData(int it, double ev, int step, double time){
        iterationNumber.add(it);
        expectedValue.add(ev);
        maxSteps.add(step);
        usedTime.add(time);
    }

    public void PrintOut(){
        System.out.println("\n");
        for (int i = 0; i < iterationNumber.size(); i++){
            System.out.println(iterationNumber.get(i) + ", " + expectedValue.get(i) +
                    ", " + maxSteps.get(i) + ", " + usedTime.get(i));
        }
    }

    public void WriteFile(String pathName){
        FileWriter fileWriter = null;
        try{
            fileWriter = new FileWriter(pathName);
            fileWriter.append("iterations,expected value,max steps,time\n");

            for (int i = 0; i < iterationNumber.size(); i++){

                fileWriter.append(String.valueOf(iterationNumber.get(i)));
                fileWriter.append(COMMA_DELIMITER);
                fileWriter.append(String.valueOf(expectedValue.get(i)));
                fileWriter.append(COMMA_DELIMITER);
                fileWriter.append(String.valueOf(maxSteps.get(i)));
                fileWriter.append(COMMA_DELIMITER);
                fileWriter.append(String.valueOf(usedTime.get(i)));

                fileWriter.append(NEW_LINE_SEPARATOR);
            }

        }
        catch (Exception e){
            System.out.println("Error while writing CSV");
            e.printStackTrace();
        }
        finally {
            try{
                fileWriter.flush();
                fileWriter.close();
            }
            catch (Exception e){
                System.out.println("Error while flushing or closing CSV");
                e.printStackTrace();
            }
        }
    }

}
