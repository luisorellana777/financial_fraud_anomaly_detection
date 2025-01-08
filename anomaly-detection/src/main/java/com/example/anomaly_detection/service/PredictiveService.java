package com.example.anomaly_detection.service;

import com.example.anomaly_detection.common.ModelMapper;
import com.example.anomaly_detection.predictor.ModelReaderService;
import com.example.anomaly_detection.predictor.impl.ModelReaderCommandLineImpl;
import com.example.anomaly_detection.model.Transaction;
import lombok.AllArgsConstructor;
import org.springframework.stereotype.Service;

@Service
@AllArgsConstructor
public class PredictiveService {

    private ModelReaderService modelReader;

    public boolean isFraud(Transaction transaction){

        //Transform transaction into data that model can understand
        return modelReader.getResult(transaction) == 1 ? true : false;
    }

}
