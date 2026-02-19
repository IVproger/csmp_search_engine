FROM nvcr.io/nvidia/tritonserver:24.10-py3

WORKDIR /opt/tritonserver

ENTRYPOINT ["tritonserver"]
CMD ["--model-repository=/models", "--grpc-port=8001", "--http-port=8002", "--metrics-port=8003", "--strict-readiness=true", "--allow-gpu-metrics=false"]