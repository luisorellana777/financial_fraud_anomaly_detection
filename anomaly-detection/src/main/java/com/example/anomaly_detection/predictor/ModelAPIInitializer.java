package com.example.anomaly_detection.predictor;

import jakarta.annotation.PreDestroy;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Profile;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Component;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Profile("rest")
@Component
public class ModelAPIInitializer {

    private String pid1;
    private String pid2;
    @Value("${model.path}")
    private String modelPath;

    @Async
    public void runModelCommandBase(){

        try {

            // Define the command to start uvicorn
            String[] command = {
                    "/bin/bash",
                    modelPath.concat("run_fastapi_service.sh"),
                    modelPath
            };

            ProcessBuilder processBuilder = new ProcessBuilder(command);
            processBuilder.redirectErrorStream(true);

            Process process = processBuilder.start();

            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    if(line.contains("process [")){
                        Matcher matcher = Pattern.compile("\\d+").matcher(line);

                        if (matcher.find()) {
                            String pid = matcher.group();
                            if(pid1==null){
                                pid1 = pid;
                            }else{
                                pid2 = pid;
                            }
                        }
                    }
                    System.out.println("[FastAPI] " + line);
                }
            }

            int exitCode = process.waitFor();
            if (exitCode != 0) {
                System.err.println("FastAPI server exited with code " + exitCode);
            }
        } catch (Exception e) {
            System.err.println("Failed to start FastAPI service.");
            throw new RuntimeException(e);
        }
    }

    @PreDestroy
    public void onShutdown() {
        try {
            Runtime.getRuntime().exec("kill -9 " + pid1);
            Runtime.getRuntime().exec("kill -9 " + pid2);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

}
