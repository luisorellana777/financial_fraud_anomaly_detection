package com.example.anomaly_detection;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableAsync;

@EnableAsync
@SpringBootApplication
public class AnomalyDetectionApplication {

	public static void main(String[] args) {
		SpringApplication.run(AnomalyDetectionApplication.class, args);
	}

}
