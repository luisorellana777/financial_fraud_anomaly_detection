package com.example.anomaly_detection.model;

import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

@Getter
@Builder
public class ResponseFraud {

    private FraudStatus status;
}
