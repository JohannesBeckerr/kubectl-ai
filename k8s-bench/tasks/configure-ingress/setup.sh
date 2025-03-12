#!/bin/bash
kubectl delete namespace ingress-test --ignore-not-found
kubectl create namespace ingress-test

# Create deployment and service
kubectl create deployment web-app --image=nginx -n ingress-test
kubectl expose deployment web-app --name=web-service --port=80 -n ingress-test
# Wait for deployment to be ready
kubectl wait --for=condition=Available --timeout=30s deployment/web-app -n ingress-test || exit 1