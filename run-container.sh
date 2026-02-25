# 1. Start the container in the background
docker-compose pull
docker-compose up -d

echo "Connecting to container..."

# 2. Enter the container
# When you type 'exit' or press Ctrl+D, the script continues to the next line
docker exec -it onnxruntime-on-qualcomm-hexagon-qcs6490 bash

# 3. Cleanup after exiting the bash session
echo "Exited container. Cleaning up..."
docker-compose down
