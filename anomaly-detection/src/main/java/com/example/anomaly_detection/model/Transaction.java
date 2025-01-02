package com.example.anomaly_detection.model;

import lombok.Data;
import lombok.Getter;
import org.springframework.web.bind.annotation.GetMapping;

@Data
public class Transaction {

    private int step;
    private Type type;
    private long amount;

    @Getter
    public enum Type{
        CASH_IN(0), CASH_OUT(1), DEBIT(2), PAYMENT(3), TRANSFER(4);
        private int code;

        Type(int code) {
            this.code = code;
        }
    }
}
