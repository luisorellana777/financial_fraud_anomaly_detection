package com.example.anomaly_detection.configuration;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Scope;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Configuration
public class PySVMBuilderConfig {

    @Scope(value = "prototype")
    @Bean(name = "pyCommandBase")
    public List<String> buildCommandBase(){

        return new ArrayList<String>(Arrays.asList("python", "/Users/luisorellanaaltamirano/Documents/Machine_Learning/anomaly-detection/src/main/resources/model/load_model_command_line.py"));
    }
}
