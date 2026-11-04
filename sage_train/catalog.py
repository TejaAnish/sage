"""Crop/disease selection: the single source of truth for dataset, prompts and training.

Why curated `pathogen`/`type`/`cue` values instead of SAGE's metadata columns: those
columns are missing for many selected classes and sometimes wrong (e.g. Rice Bacterial
Leaf Blight carries Burkholderia glumae, the panicle-blight pathogen).
See reports/metadata_audit.md.
"""

HF_SOURCE_DATASET = "tirtho149/SAGE"

# Numeric raw labels ("A2H0H0R1Plantdisease/35") cannot be audited, and that source
# overlaps the PlantVillage-derived classes that have readable labels.
EXCLUDE_SOURCES = {"A2H0H0R1Plan"}

CROPS = {
    "rice": {
        "dataset_crop": "Rice",
        "display": "Rice",
        "exclude_raw_labels": {"Rice/Bacterial_Leaf_Streak"},  # different disease merged in by SAGE
        "diseases": {
            "Blast": {
                "pathogen": "Magnaporthe oryzae", "type": "Fungal",
                "cue": "spindle-shaped lesions with grey centres and brown margins; neck or node rot",
            },
            "Brown Spot": {
                "pathogen": "Bipolaris oryzae", "type": "Fungal",
                "cue": "oval brown spots with grey centres, scattered over the leaf like sesame seeds",
            },
            "Bacterial Leaf Blight": {
                "pathogen": "Xanthomonas oryzae pv. oryzae", "type": "Bacterial",
                "cue": "yellow to straw-coloured wavy stripes spreading down from leaf tips and margins",
            },
            "Tungro": {
                "pathogen": "Rice tungro bacilliform and spherical viruses", "type": "Viral",
                "cue": "yellow to orange leaf discolouration starting at the tip, on stunted plants",
            },
        },
    },
    "corn": {
        "dataset_crop": "Corn",  # SAGE has no separate Maize crop: this model serves Corn and Maize
        "display": "Corn (Maize)",
        "exclude_raw_labels": set(),
        "diseases": {
            "Common Rust": {
                "pathogen": "Puccinia sorghi", "type": "Fungal",
                "cue": "powdery brick-red pustules scattered on both leaf surfaces",
            },
            "Northern Leaf Blight": {
                "pathogen": "Exserohilum turcicum", "type": "Fungal",
                "cue": "long cigar-shaped grey to tan lesions on the leaves",
            },
            "Gray Leaf Spot": {
                "pathogen": "Cercospora zeae-maydis", "type": "Fungal",
                "cue": "narrow rectangular grey to tan lesions limited by the leaf veins",
            },
            "Southern Rust": {
                "pathogen": "Puccinia polysora", "type": "Fungal",
                "cue": "dense small orange circular pustules, mainly on the upper leaf surface",
            },
            "Smut": {
                "pathogen": "Ustilago maydis", "type": "Fungal",
                "cue": "swollen white to grey galls that burst into dark sooty spore masses",
            },
        },
    },
    "soybean": {
        "dataset_crop": "Soybean",
        "display": "Soybean",
        "exclude_raw_labels": set(),
        "diseases": {
            "Frogeye Leaf Spot": {
                "pathogen": "Cercospora sojina", "type": "Fungal",
                "cue": "small circular grey lesions with distinct reddish-brown borders",
            },
            "Bacterial Blight": {
                "pathogen": "Pseudomonas savastanoi pv. glycinea", "type": "Bacterial",
                "cue": "angular water-soaked spots turning brown with yellow halos, leaves look tattered",
            },
            "Bacterial Pustule": {
                "pathogen": "Xanthomonas axonopodis pv. glycines", "type": "Bacterial",
                "cue": "small pale-green spots with raised light-coloured pustules at their centres",
            },
            "Brown Spot": {
                "pathogen": "Septoria glycines", "type": "Fungal",
                "cue": "small irregular dark-brown spots on lower leaves with yellowing around them",
            },
            "Rust": {
                "pathogen": "Phakopsora pachyrhizi", "type": "Fungal",
                "cue": "small tan to reddish-brown angular lesions with pustules on the leaf underside",
            },
        },
    },
    "tomato": {
        "dataset_crop": "Tomato",
        "display": "Tomato",
        "exclude_raw_labels": set(),
        "diseases": {
            "Tomato Leaf Curl Virus": {
                "pathogen": "Tomato leaf curl begomoviruses", "type": "Viral",
                "cue": "upward-curled, puckered, yellow-edged small leaves on stunted plants",
            },
            "Early Blight": {
                "pathogen": "Alternaria solani / A. linariae", "type": "Fungal",
                "cue": "dark spots with concentric target-like rings, starting on lower leaves",
            },
            "Late Blight": {
                "pathogen": "Phytophthora infestans", "type": "Oomycete",
                "cue": "large irregular water-soaked dark blotches, sometimes with white growth beneath",
            },
            "Septoria Leaf Blotch": {
                "pathogen": "Septoria lycopersici", "type": "Fungal",
                "cue": "many small circular spots with grey centres, dark margins and tiny black dots",
            },
            "Bacterial Leaf Spot": {
                "pathogen": "Xanthomonas spp.", "type": "Bacterial",
                "cue": "small dark greasy angular spots, sometimes with yellow halos",
            },
        },
    },
    "wheat": {
        "dataset_crop": "Wheat",
        "display": "Wheat",
        # Bunts infect grain and look unlike loose smut; SAGE merges them into "Smut".
        "exclude_raw_labels": {"Wheat/Karnal_Bunt", "Wheat/Dwarf_Bunt"},
        "diseases": {
            "Rust": {
                "pathogen": "Puccinia spp. (stripe, leaf and stem rust)", "type": "Fungal",
                "cue": "yellow, orange or reddish-brown powdery pustules in stripes or scattered",
            },
            "Smut": {
                "pathogen": "Ustilago tritici", "type": "Fungal",
                "cue": "ears replaced by black powdery spore masses",
            },
            "Septoria Leaf Blotch": {
                "pathogen": "Zymoseptoria tritici", "type": "Fungal",
                "cue": "irregular tan blotches containing tiny black fruiting dots",
            },
            "Powdery Mildew": {
                "pathogen": "Blumeria graminis f. sp. tritici", "type": "Fungal",
                "cue": "white to grey fluffy powdery patches on leaves and stems",
            },
            "Head Scab": {
                "pathogen": "Fusarium graminearum", "type": "Fungal",
                "cue": "bleached straw-coloured spikelets on otherwise green ears",
            },
        },
    },
}

# Frontend names resolved to a model key; Corn and Maize share one model.
CROP_ALIASES = {"maize": "corn", "paddy": "rice"}


def resolve_crop(name):
    key = name.strip().lower()
    key = CROP_ALIASES.get(key, key)
    if key not in CROPS:
        raise KeyError(f"Unknown crop {name!r}; expected one of {sorted(CROPS)} or {sorted(CROP_ALIASES)}")
    return key


def disease_names(crop_key):
    return list(CROPS[crop_key]["diseases"])
