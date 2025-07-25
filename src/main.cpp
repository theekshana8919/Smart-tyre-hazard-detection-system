#include <WiFi.h>
#include <PubSubClient.h>

// Pins
#define NAIL_SENSOR_PIN 14      // Change if needed
#define DEBOUNCE_DELAY 1000     // 1 sec

// WiFi credentials
const char *ssid = "Theekshana";
const char *password = "12341234";

// MQTT broker and topic
const char *mqtt_server = "test.mosquitto.org";
const char *topic = "tyre/alerts";

WiFiClient espClient;
PubSubClient client(espClient);

// Connect to WiFi
void setup_wifi() {
  delay(10);
  Serial.begin(9600);
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);

  WiFi.begin(ssid, password);

  int retryCount = 0;
  while (WiFi.status() != WL_CONNECTED && retryCount < 20) {
    delay(500);
    Serial.print(".");
    retryCount++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected. IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\nWiFi failed!");
  }
}

// Connect to MQTT broker
void reconnect() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    if (client.connect("ESP32NailDetector")) {
      Serial.println("connected to MQTT");
    } else {
      Serial.print(" failed, rc=");
      Serial.print(client.state());
      Serial.println(" retrying in 5s");
      delay(5000);
    }
  }
}

void setup() {
  pinMode(NAIL_SENSOR_PIN, INPUT);
  setup_wifi();
  client.setServer(mqtt_server, 1883);
}

void loop() {
  static bool lastState = false;
  static unsigned long lastDetectionTime = 0;

  if (!client.connected()) {
    reconnect();
  }
  client.loop();

  bool currentState = digitalRead(NAIL_SENSOR_PIN);
  unsigned long currentTime = millis();

  // DEBUG: print sensor value
  Serial.print("Sensor: ");
  Serial.println(currentState);

  if (currentState && !lastState && (currentTime - lastDetectionTime > DEBOUNCE_DELAY)) {
    Serial.println("Nail detected!");

    bool success = client.publish(topic, "HIGH");
    if (success) {
      Serial.println("MQTT publish success: HIGH");
    } else {
      Serial.println("MQTT publish FAILED");
    }

    lastDetectionTime = currentTime;
  }

  lastState = currentState;
  delay(10);
}
