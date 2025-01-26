package com.example.anomaly_detection.configuration;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Scope;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Configuration
public class PySVMBuilderConfig {

    @Value("${model.path}")
    private String modelPath;

    @Scope(value = "prototype")
    @Bean(name = "pyCommandBase")
    public List<String> buildCommandBase(){

        return new ArrayList<String>(Arrays.asList("python", modelPath.concat("load_model_command_line.py"), modelPath));
    }
}
