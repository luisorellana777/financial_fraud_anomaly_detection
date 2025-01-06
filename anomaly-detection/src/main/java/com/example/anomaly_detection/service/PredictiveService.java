package com.example.anomaly_detection.service;

import com.example.anomaly_detection.common.ModelMapper;
import com.example.anomaly_detection.common.ModelReader;
import com.example.anomaly_detection.model.Transaction;
import lombok.AllArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.Arrays;
import java.util.List;

@Service
@AllArgsConstructor
public class PredictiveService {

    private ModelReader modelReader;
    private ModelMapper modelMapper;

    public boolean isFraud(Transaction transaction){

        List<String> inputMapped = modelMapper.map(transaction);

        //Transform transaction into data that model can understand
        return modelReader.getResult(inputMapped) == 1 ? true : false;
    }

}
