FROM openjdk:17-slim

# Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3 and necessary dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the Python requirements file into the container
COPY build/resources/main/model/requirements.txt .

# Install Python dependencies from requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

WORKDIR /app
VOLUME /tmp
EXPOSE 8080
COPY src/main/resources/model /app/resources/model
COPY build/libs/anomaly-detection-0.0.1-SNAPSHOT.jar /app/app.jar
ENV JAVA_OPTS=""
CMD ["java", "-jar", "app.jar"]
#CMD ["ls", "-l", "/app/resources/model/"]