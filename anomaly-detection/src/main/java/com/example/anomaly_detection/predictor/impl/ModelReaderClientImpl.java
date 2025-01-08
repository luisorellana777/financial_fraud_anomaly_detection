package com.example.anomaly_detection.predictor.impl;

import com.example.anomaly_detection.model.Transaction;
import com.example.anomaly_detection.predictor.ModelAPIInitializer;
import com.example.anomaly_detection.predictor.ModelReaderService;
import jakarta.annotation.PostConstruct;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.beans.factory.config.BeanPostProcessor;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

@Profile("rest")
@Component
public class ModelReaderClientImpl implements ModelReaderService {

    private final WebClient webClient;
    private ModelAPIInitializer modelAPIInitializer;

    public ModelReaderClientImpl(WebClient.Builder webClientBuilder, ModelAPIInitializer modelAPIInitializer, @Value("${model.url.base}") String base, @Value("${model.url.port}") String port, @Value("${model.url.endpoint}") String endpoint) {
        this.webClient = webClientBuilder.baseUrl(base.concat(":").concat(port).concat(endpoint)).build();
        this.modelAPIInitializer = modelAPIInitializer;
    }

    @PostConstruct
    public void initializeFastAPIPostConstruct() {
        modelAPIInitializer.runModelCommandBase();
    }

    public int getResult(Transaction transaction){

        Mono<Integer> map = webClient.post()
                .bodyValue(transaction)
                .retrieve()
                .bodyToMono(String.class)
                .map(value -> Integer.parseInt(value.replace('[', ' ').replace(']', ' ').trim()));

        return map.block();
    }
}
