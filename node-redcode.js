
[
    {
        "id": "a1b2c3d4.123456",
        "type": "mqtt in",
        "z": "7890efgh.567890",
        "name": "ESP32 Raw Data",
        "topic": "tyre/raw",
        "qos": "1",
        "datatype": "auto",
        "broker": "abc12345.67890",
        "x": 150,
        "y": 100,
        "wires": [
            ["b2c3d4e5.234567"]
        ]
    },
    {
        "id": "b2c3d4e5.234567",
        "type": "function",
        "z": "7890efgh.567890",
        "name": "Process Hazard Data",
        "func": "// Analyze sensor data for patterns indicating nails\nconst threshold = 3; // Number of consecutive detections to confirm\n\n// Initialize context variables if they don't exist\ncontext.detectionCount = context.detectionCount || 0;\ncontext.lastDetection = context.lastDetection || 0;\n\nconst currentTime = Date.now();\nconst payload = msg.payload;\n\n// Reset if too much time passed since last detection\nif (currentTime - context.lastDetection > 5000) {\n    context.detectionCount = 0;\n}\n\n// Check for nail detection signal\nif (payload === \"1\") {\n    context.detectionCount++;\n    context.lastDetection = currentTime;\n    \n    // Only confirm detection after threshold is reached\n    if (context.detectionCount >= threshold) {\n        msg.payload = {\n            type: \"nail\",\n            timestamp: new Date().toISOString(),\n            confidence: Math.min(100, context.detectionCount * 20)\n        };\n        context.detectionCount = 0; // Reset after confirmation\n        return msg;\n    }\n} else if (payload === \"0\") {\n    // Publish clear signal when hazard is gone\n    msg.payload = {\n        type: \"clear\",\n        timestamp: new Date().toISOString()\n    };\n    return msg;\n}\n\n// No output unless we have a confirmed detection\nreturn null;",
        "outputs": 1,
        "noerr": 0,
        "initialize": "",
        "finalize": "",
        "libs": [],
        "x": 350,
        "y": 100,
        "wires": [
            ["c3d4e5f6.345678", "d4e5f6g7.456789"]
        ]
    },
    {
        "id": "c3d4e5f6.345678",
        "type": "mqtt out",
        "z": "7890efgh.567890",
        "name": "App Alerts",
        "topic": "tyre/alerts",
        "qos": "1",
        "retain": "false",
        "broker": "abc12345.67890",
        "x": 550,
        "y": 100,
        "wires": []
    },
    {
        "id": "d4e5f6g7.456789",
        "type": "debug",
        "z": "7890efgh.567890",
        "name": "Debug Output",
        "active": true,
        "tosidebar": true,
        "console": false,
        "tostatus": false,
        "complete": "false",
        "statusVal": "",
        "statusType": "auto",
        "x": 550,
        "y": 200,
        "wires": []
    },
    {
        "id": "abc12345.67890",
        "type": "mqtt-broker",
        "name": "MQTT Broker",
        "broker": "test.mosquitto.org",
        "port": "1883",
        "clientid": "",
        "autoConnect": true,
        "usetls": false,
        "protocolVersion": "4",
        "keepalive": "60",
        "cleansession": true,
        "birthTopic": "",
        "birthQos": "0",
        "birthPayload": "",
        "birthMsg": {},
        "closeTopic": "",
        "closeQos": "0",
        "closePayload": "",
        "closeMsg": {},
        "willTopic": "",
        "willQos": "0",
        "willPayload": "",
        "willMsg": {},
        "sessionExpiry": ""
    }
]
