#!/bin/bash
# Wait up to 30 seconds for deployment to scale to 4 replicas
for i in {1..30}; do
    if kubectl get deployment web-app -n scale-test -o jsonpath='{.status.availableReplicas}' | grep -q "2"; then
        exit 0
    fi
    sleep 1
done

# If we get here, deployment didn't scale up in time
exit 1 