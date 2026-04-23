import time
import torch
from gliner2 import GLiNER2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

model = GLiNER2.from_pretrained("fastino/gliner2-multi-v1")
model.to(args.device)
model.eval()

text = "Il signor Mario Rossi vive a Roma e lavora per Semplifica s.r.l. dal 2020. L'azienda, fondata da Giuseppe Verdi, ha recentemente aperto una nuova sede a Milano, vicino al Duomo. Nel 2023, il fatturato è cresciuto del 45%, spinto dalle nuove tecnologie di intelligenza artificiale. La dottoressa Francesca Bianchi, CEO della divisione europea, ha tenuto una conferenza a Parigi il 15 Maggio 2024, annunciando partnership strategiche con Microsoft e Google."
labels = ['person', 'organization', 'location', 'date', 'time', 'event', 'facility', 'product', 'language', 'law', 'percent', 'quantity', 'money', 'ordinal', 'cardinal', 'nationality', 'religion', 'title', 'profession', 'country', 'city', 'state', 'brand', 'vehicle', 'weapon', 'disease', 'drug', 'chemical', 'material', 'color', 'shape', 'animal', 'plant', 'food', 'beverage', 'sport', 'game', 'award', 'art', 'book', 'movie', 'song', 'music', 'software', 'website', 'company', 'university', 'school', 'hospital', 'airport', 'station']

# Warmup
for _ in range(5):
    # In GLiNER2 extract_entities API is a bit different? Let's check predict API.
    # Usually it's model.extract_entities(text, labels, threshold=0.15)
    _ = model.extract_entities(text, labels, threshold=0.15)

start_time = time.time()
num_runs = 50
total_entities = 0

for _ in range(num_runs):
    entities = model.extract_entities(text, labels, threshold=0.15)
    total_entities += sum(len(v) for v in entities['entities'].values())

total_time = time.time() - start_time
time_per_run_ms = (total_time / num_runs) * 1000
time_per_sentence_ms = time_per_run_ms / 4.0
time_per_entity_ms = (total_time / total_entities) * 1000 if total_entities > 0 else 0

print(f"\n--- RESULTS FOR {args.device.upper()} ---")
print(f"Device: {args.device}")
print(f"Total time: {total_time:.4f}s")
print(f"Avg Time per Sentence: {time_per_sentence_ms:.4f} ms")
print(f"Time per run: {time_per_run_ms:.4f} ms")
print(f"Time per entity: {time_per_entity_ms:.4f} ms")
print(f"Total entities extracted per run: {total_entities / num_runs:.2f}")

