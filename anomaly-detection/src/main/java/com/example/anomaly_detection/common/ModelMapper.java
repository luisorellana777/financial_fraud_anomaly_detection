package com.example.anomaly_detection.common;

import com.example.anomaly_detection.model.Transaction;
import org.springframework.stereotype.Component;

import java.util.Arrays;
import java.util.List;

@Component
public class ModelMapper {
    double minStep = 1.000000;
    double maxStep = 743.000000;
    double minType = 0.000000;
    double maxType = 4.000000;
    double minAmount = 0.000000;
    double maxAmount = 92445516.640000;

    //reverse_normalized_value = normalized_value * (max(x) - min(x)) + min(x)
    //df = (df - df.min()) / (df.max() - df.min())
    public List<String> map(Transaction transaction) {

        String normalizedStep = normalizeValue(transaction.getStep(), minStep, maxStep);
        String normalizedType = normalizeValue(transaction.getType().getCode(), minType, maxType);
        String normalizedAmount = normalizeValue(transaction.getAmount(), minAmount, maxAmount);

        return Arrays.asList(normalizedStep, normalizedType, normalizedAmount);
    }

    private String normalizeValue(long value, double minValue, double maxValue){
        double normalized = (value - minValue) / (maxValue - minValue);
        return String.valueOf(normalized);
    }
}
