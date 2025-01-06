package com.example.anomaly_detection.common;

import lombok.AllArgsConstructor;
import org.springframework.stereotype.Component;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

@AllArgsConstructor
@Component
public class ModelReader {

    private List<String> pyCommandBase;

    public int getResult(List<String> values){

        try {

            // Run the Python script
            ProcessBuilder pb = new ProcessBuilder(Stream.concat(pyCommandBase.stream(), values.stream()).collect(Collectors.toList()));
            Process process = pb.start();

            // Read Python script output
            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream())
            );
            String line;
            if ((line = reader.readLine()) != null) {
                return Integer.parseInt(line.replace('[', ' ').replace(']', ' ').trim());
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
        return -1;
    }
}
