package com.example.anomaly_detection.predictor.impl;

import com.example.anomaly_detection.common.ModelMapper;
import com.example.anomaly_detection.model.Transaction;
import com.example.anomaly_detection.predictor.ModelReaderService;
import lombok.AllArgsConstructor;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

@Profile("command")
@Component
@AllArgsConstructor
public class ModelReaderCommandLineImpl implements ModelReaderService {

    private List<String> pyCommandBase;

    private ModelMapper modelMapper;

    public int getResult(Transaction transaction){

        List<String> inputMapped = modelMapper.map(transaction);

        try {

            ProcessBuilder pb = new ProcessBuilder(Stream.concat(pyCommandBase.stream(), inputMapped.stream()).collect(Collectors.toList()));
            Process process = pb.start();

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
