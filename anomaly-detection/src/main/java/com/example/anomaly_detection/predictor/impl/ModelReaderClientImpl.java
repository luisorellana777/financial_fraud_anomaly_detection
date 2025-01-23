package com.example.anomaly_detection.predictor.impl;

import com.example.anomaly_detection.configuration.ModelRestConfig;
import com.example.anomaly_detection.model.Transaction;
import com.example.anomaly_detection.predictor.ModelAPIInitializer;
import com.example.anomaly_detection.predictor.ModelReaderService;
import jakarta.annotation.PostConstruct;
import lombok.AllArgsConstructor;
import org.springframework.context.annotation.Profile;
import org.springframework.http.*;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;

@Profile("rest")
@Component
@AllArgsConstructor
public class ModelReaderClientImpl implements ModelReaderService {

    private RestTemplate restTemplate;

    private ModelRestConfig.ModelURL modelURL;
    private ModelAPIInitializer modelAPIInitializer;

    @PostConstruct
    public void initializeFastAPIPostConstruct() {
        modelAPIInitializer.runModelCommandBase();
    }

    public int getResult(Transaction transaction){

        HttpHeaders headers = new HttpHeaders();
        headers.set("Content-Type", "application/json");

        ResponseEntity<Integer> response = restTemplate.exchange(
                modelURL.getBase().concat(":").concat(modelURL.getPort()).concat(modelURL.getEndpoint()),
                HttpMethod.POST,
                new HttpEntity<>(transaction, headers),
                Integer.class
        );
        return response.getBody();

    }
}
