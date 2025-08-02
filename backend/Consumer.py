from confluent_kafka import Consumer
import json

# ---------- Kafka Consumer Configuration ----------
conf = {
    'bootstrap.servers': 'localhost:9092',     
    'group.id': 'well-stream-consumer-group',  
    'auto.offset.reset': 'earliest'            
}

topic = 'fdms.well.stream'

# ---------- Create Kafka Consumer ----------
consumer = Consumer(conf)
consumer.subscribe([topic])

print(f"📡 Consumer connected to topic '{topic}' and waiting for data... (Press Ctrl+C to stop)")

try:
    while True:
        msg = consumer.poll(1.0) 

        if msg is None:
            continue
        if msg.error():
            print(f"⚠️ Error: {msg.error()}")
            continue

        # Decode message
        key = msg.key().decode('utf-8') if msg.key() else None
        value = json.loads(msg.value().decode('utf-8'))

        # Print essential data
        print("─────────────────────────────────────────────")
        print(f"📦 Record ID: {value.get('Record_ID')} | Phase: {value.get('Phase_Operation')}")
        print(f"🕒 DateTime: {value.get('DateTime')}")
        print(f"🌍 Location: ({value.get('LAT'):.4f}, {value.get('LONG'):.4f}) | Formation: {value.get('Formation_Type')}")
        print(f"🌡 Temp: {value.get('Reservoir_Temperature'):.2f} °F | Pressure: {value.get('Pressure_Reservoir'):.2f} psi")
        print(f"💧 Mud Type: {value.get('Mud_Type')} | pH: {value.get('Mud_pH'):.2f} | Viscosity: {value.get('Viscosity'):.2f}")
        print(f"🧱 Clay: {value.get('Clay_Mineralogy_Type')} | Content: {value.get('Clay_Content_Percent'):.2f} %")
        print("─────────────────────────────────────────────")

except KeyboardInterrupt:
    print("\n⛔️ Stopped by user.")

finally:
    consumer.close()

