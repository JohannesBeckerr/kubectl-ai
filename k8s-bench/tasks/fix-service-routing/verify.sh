#!/bin/bash
# Wait for service endpoints to be ready
if kubectl wait --for=condition=Available --timeout=30s service/nginx -n web; then
    # Check if service has endpoints
    endpoints=$(kubectl get endpoints nginx -n web -o jsonpath='{.subsets[0].addresses}')
    if [[ ! -z "$endpoints" ]]; then
        # Verify service can access the pod
        if kubectl run -n web test-connection --image=busybox --restart=Never --rm -i --wait --timeout=10s \
            -- wget -qO- nginx; then
            exit 0
        fi
    fi
fi

# If we get here, service connection failed
exit 1 