#!/bin/bash
# Wait up to 30 seconds for deployment to scale down to 2 replicas
for i in {1..30}; do
    if kubectl get deployment web-service -n scale-down-test -o jsonpath='{.status.availableReplicas}' | grep -q "2"; then
        exit 0
    fi
    sleep 1
done

# If we get here, deployment didn't scale down in time
exit 1 