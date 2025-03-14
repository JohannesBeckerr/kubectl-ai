#!/bin/bash

# Cleanup existing namespaces if they exist
kubectl delete namespace ns1 --ignore-not-found
kubectl delete namespace ns2 --ignore-not-found

# Wait for namespaces to be fully deleted
echo "Waiting for namespaces to be fully deleted..."
while kubectl get namespace ns1 2>/dev/null || kubectl get namespace ns2 2>/dev/null; do
    sleep 1
done

# Create the namespaces
kubectl create namespace ns1
kubectl create namespace ns2

# Enable NetworkPolicy in both namespaces
kubectl label namespace ns1 network-policy=enabled --overwrite
kubectl label namespace ns2 network-policy=enabled --overwrite

# Deploy httpd pods in each namespace for testing connectivity
kubectl run httpd-ns1 -n ns1 --image=httpd:alpine
kubectl run httpd-ns2 -n ns2 --image=httpd:alpine

# Deploy test pods with curl for testing connectivity
kubectl run curl-ns1 -n ns1 --image=curlimages/curl --command -- sleep 3600
kubectl run curl-ns2 -n ns2 --image=curlimages/curl --command -- sleep 3600

# Wait for pods to be ready
echo "Waiting for pods to be ready..."
kubectl wait --for=condition=Ready pod/httpd-ns1 -n ns1 --timeout=60s
kubectl wait --for=condition=Ready pod/httpd-ns2 -n ns2 --timeout=60s
kubectl wait --for=condition=Ready pod/curl-ns1 -n ns1 --timeout=60s
kubectl wait --for=condition=Ready pod/curl-ns2 -n ns2 --timeout=60s

echo "Setup completed" 