# SAGE metadata audit

Rows: 820,504 | crops: 335 | disease labels: 526 | sources: 32
Unique on (raw_label, filename): True | unique on filename alone: False
Rows with a pathogen: 55.2%. Sources excluded here: ['A2H0H0R1Plan'].
Counts are before perceptual de-duplication, so real class sizes will be lower. `build_dataset.py finalize` publishes the final counts in each crop's manifest.

## Rice

| disease | rows | sources | largest_source | largest_share | fetched |
|---|---|---|---|---|---|
| Bacterial Leaf Blight | 4316 | 4 | BDCropVegeta | 48% | 4316 |
| Blast | 7411 | 6 | BDCropVegeta | 43% | 5000 |
| Brown Spot | 5495 | 5 | BDCropVegeta | 47% | 5000 |
| Tungro | 4412 | 3 | BDCropVegeta | 54% | 4412 |

## Corn (Maize)

| disease | rows | sources | largest_source | largest_share | fetched |
|---|---|---|---|---|---|
| Common Rust | 16911 | 12 | RadyPlantDis | 30% | 5000 |
| Gray Leaf Spot | 8065 | 10 | AvinashPlant | 26% | 5000 |
| Northern Leaf Blight | 7809 | 8 | LeafNet | 30% | 5000 |
| Smut | 505 | 3 | Corn_Disease | 48% | 505 |
| Southern Rust | 309 | 2 | Corn_Disease | 95% | 309 |

## Soybean

| disease | rows | sources | largest_source | largest_share | fetched |
|---|---|---|---|---|---|
| Bacterial Blight | 6336 | 5 | Soybean_PNAS | 94% | 5000 |
| Bacterial Pustule | 6112 | 3 | Soybean_PNAS | 96% | 5000 |
| Brown Spot | 6125 | 4 | Soybean_PNAS | 96% | 5000 |
| Frogeye Leaf Spot | 6458 | 4 | Soybean_PNAS | 92% | 5000 |
| Rust | 400 | 4 | Soybean_Dise | 46% | 400 |

## Tomato

| disease | rows | sources | largest_source | largest_share | fetched |
|---|---|---|---|---|---|
| Bacterial Leaf Spot | 20824 | 10 | CDDM | 28% | 5000 |
| Early Blight | 17584 | 11 | CDDM | 24% | 5000 |
| Late Blight | 17742 | 10 | CDDM | 33% | 5000 |
| Septoria Leaf Blotch | 28314 | 10 | RadyPlantDis | 35% | 5000 |
| Tomato Leaf Curl Virus | 33956 | 11 | CDDM | 23% | 5000 |

## Wheat

| disease | rows | sources | largest_source | largest_share | fetched |
|---|---|---|---|---|---|
| Head Scab | 365 | 3 | PlantWild | 87% | 365 |
| Powdery Mildew | 318 | 3 | PlantWild | 85% | 318 |
| Rust | 5579 | 6 | CDDM | 46% | 5000 |
| Septoria Leaf Blotch | 577 | 6 | PlantWild | 40% | 577 |
| Smut | 1186 | 4 | CDDM | 79% | 1186 |
