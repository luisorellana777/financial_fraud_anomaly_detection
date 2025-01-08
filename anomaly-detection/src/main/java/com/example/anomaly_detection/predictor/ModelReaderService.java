package com.example.anomaly_detection.predictor;

import com.example.anomaly_detection.model.Transaction;

import java.util.List;

public interface ModelReaderService {

    int getResult(Transaction transaction);
}
