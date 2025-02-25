#!/bin/bash
# Create namespace and deploy a test application
kubectl create namespace ingress-test

# Create deployment and service
kubectl create deployment web-app --image=nginx -n ingress-test
kubectl expose deployment web-app --name=web-service --port=80 -n ingress-test

# Wait for deployment to be ready
for i in {1..30}; do
    if kubectl get deployment web-app -n ingress-test -o jsonpath='{.status.availableReplicas}' | grep -q "1"; then
        exit 0
    fi
    sleep 1
done 