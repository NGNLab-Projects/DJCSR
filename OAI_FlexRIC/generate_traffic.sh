#!/bin/bash
echo "Starting continuous traffic for connected UEs in namespaces..."

for i in {1..10}; do
    # Check if namespace ue$i exists
    if ip netns list | grep -q "ue$i"; then
        # Check if the tunnel exists in this namespace
        TUNNEL=$(sudo ip netns exec ue$i ip -o link show | awk -F': ' '{print $2}' | grep oaitun_ue)
        if [ ! -z "$TUNNEL" ]; then
            echo "Starting ping flood traffic on interface $TUNNEL in namespace ue$i..."
            sudo ip netns exec ue$i ping -I $TUNNEL 192.168.70.135 -s 1400 -f > /dev/null 2>&1 &
        else
            echo "Warning: UE $i is not fully connected yet (no oaitun_ue interface)."
        fi
    fi
done

echo ""
echo "Traffic generation running in the background!"
echo "To stop the traffic later, run: sudo killall ping"
