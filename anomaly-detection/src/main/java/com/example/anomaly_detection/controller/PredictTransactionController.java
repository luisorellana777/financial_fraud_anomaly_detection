package com.example.anomaly_detection.controller;

import com.example.anomaly_detection.model.Transaction;
import com.example.anomaly_detection.service.PredictiveService;
import lombok.AllArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

@RestController
@AllArgsConstructor
public class PredictTransactionController {

    private PredictiveService predictiveService;

    @GetMapping(value = "/transaction/fraud")
    public boolean isFraud(@RequestBody Transaction transaction){
        return predictiveService.isFraud(transaction);
    }
}
