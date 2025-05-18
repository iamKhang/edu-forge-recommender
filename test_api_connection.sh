#!/bin/bash
# Script to test API connection and diagnose issues

# Default values
API_URL=${1:-"http://eduforge-post-app:8080/api/posts/training-data/all"}
TIMEOUT=${2:-30}
VERBOSE=${3:-true}

echo "=== API Connection Test ==="
echo "URL: $API_URL"
echo "Timeout: $TIMEOUT seconds"
echo "Verbose: $VERBOSE"
echo "==========================="

# Function to print verbose output
print_verbose() {
    if [ "$VERBOSE" = true ]; then
        echo "$1"
    fi
}

# Test basic connectivity with ping
if [[ $API_URL =~ ^https?://([^:/]+) ]]; then
    HOST="${BASH_REMATCH[1]}"
    print_verbose "Testing connectivity to host: $HOST"
    
    ping -c 3 $HOST
    PING_STATUS=$?
    
    if [ $PING_STATUS -eq 0 ]; then
        echo "✅ Host $HOST is reachable"
    else
        echo "❌ Host $HOST is not reachable"
    fi
fi

# Test DNS resolution
print_verbose "Testing DNS resolution for $HOST"
host $HOST
DNS_STATUS=$?

if [ $DNS_STATUS -eq 0 ]; then
    echo "✅ DNS resolution successful for $HOST"
else
    echo "❌ DNS resolution failed for $HOST"
fi

# Test HTTP connection with curl
echo "Testing HTTP connection to $API_URL"
print_verbose "Running: curl -s -o /dev/null -w '%{http_code}' -m $TIMEOUT $API_URL"

HTTP_STATUS=$(curl -s -o /dev/null -w '%{http_code}' -m $TIMEOUT $API_URL)
CURL_STATUS=$?

if [ $CURL_STATUS -eq 0 ]; then
    echo "✅ HTTP connection successful, status code: $HTTP_STATUS"
else
    echo "❌ HTTP connection failed with curl exit code: $CURL_STATUS"
    
    if [ $CURL_STATUS -eq 28 ]; then
        echo "   Reason: Connection timed out after $TIMEOUT seconds"
    elif [ $CURL_STATUS -eq 7 ]; then
        echo "   Reason: Failed to connect to host or proxy"
    elif [ $CURL_STATUS -eq 6 ]; then
        echo "   Reason: Could not resolve host"
    else
        echo "   Reason: Unknown error"
    fi
fi

# Get response headers
if [ "$VERBOSE" = true ]; then
    echo "Getting response headers..."
    curl -s -I -m $TIMEOUT $API_URL
    
    echo "Getting response size..."
    RESPONSE_SIZE=$(curl -s -m $TIMEOUT $API_URL | wc -c)
    echo "Response size: $RESPONSE_SIZE bytes"
fi

# Test with wget as an alternative
echo "Testing with wget..."
wget --spider --timeout=$TIMEOUT -q $API_URL
WGET_STATUS=$?

if [ $WGET_STATUS -eq 0 ]; then
    echo "✅ wget connection successful"
else
    echo "❌ wget connection failed with exit code: $WGET_STATUS"
fi

# Test with Python requests if available
if command -v python3 &> /dev/null; then
    echo "Testing with Python requests..."
    python3 -c "
import requests
import sys
try:
    response = requests.get('$API_URL', timeout=$TIMEOUT)
    print(f'✅ Python requests successful, status code: {response.status_code}')
    print(f'   Response size: {len(response.text)} bytes')
    print(f'   Content type: {response.headers.get(\"Content-Type\")}')
except Exception as e:
    print(f'❌ Python requests failed: {e}')
    sys.exit(1)
"
    PYTHON_STATUS=$?
    
    if [ $PYTHON_STATUS -ne 0 ]; then
        echo "❌ Python requests test failed"
    fi
fi

echo "=== Test Complete ==="
