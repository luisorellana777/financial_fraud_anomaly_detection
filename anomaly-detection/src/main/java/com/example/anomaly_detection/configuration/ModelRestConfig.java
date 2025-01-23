package com.example.anomaly_detection.configuration;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;

@Configuration
public class ModelRestConfig {

    @Bean(name="restTemplate")
    public RestTemplate createRestTemplate(){
        return new RestTemplate();
    }

    @Getter
    @Setter
    @Component
    @ConfigurationProperties(prefix = "model.url")
    public class ModelURL{
        private String base;
        private String port;
        private String endpoint;
    }

}
