package com.example.anomaly_detection.common;

import com.example.anomaly_detection.model.Transaction;
import org.springframework.stereotype.Component;

import java.util.Arrays;
import java.util.List;

@Component
public class ModelMapper {

    //reverse_normalized_value = normalized_value * (max(x) - min(x)) + min(x)
    //df = (df - df.min()) / (df.max() - df.min())
    public List<String> map(Transaction transaction) {

        return Arrays.asList(String.valueOf(transaction.getStep()), transaction.getType().name(), String.valueOf(transaction.getAmount()));
    }

}
