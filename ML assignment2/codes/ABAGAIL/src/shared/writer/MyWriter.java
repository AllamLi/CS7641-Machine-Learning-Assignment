package shared.writer;

import shared.writer.CSVWriter;
/**
 * Created by allam on 3/10/17.
 */
public class MyWriter {
    private CSVWriter writer;
    public MyWriter(String fileName, String st[]){
        writer = new CSVWriter("csv/" + fileName, st);
    }

    public void open(){
        try{
            writer.open();
        }
        catch (Exception e){
            e.printStackTrace();
        }
    }

    public void write(String st){
        try{
            writer.write(st);
        }
        catch (Exception e){
            e.printStackTrace();
        }
    }

    public void finishRow(){
        try{
            writer.nextRecord();
        }
        catch (Exception e){
            e.printStackTrace();
        }
    }

    public void close(){
        try{
            writer.close();
        }
        catch (Exception e){
            e.printStackTrace();
        }
    }
}
