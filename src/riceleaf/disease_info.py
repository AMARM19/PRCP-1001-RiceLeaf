"""Short agronomic notes shown alongside predictions in the UI."""

DISEASE_INFO = {
    "Bacterial leaf blight": {
        "pathogen": "Xanthomonas oryzae pv. oryzae (bacterium)",
        "symptoms": "Water-soaked streaks from the leaf tip/edges that turn yellow then "
                    "greyish-white; long wavy lesions along the leaf margin.",
        "management": [
            "Grow resistant varieties (e.g. carrying Xa21 / xa13 genes).",
            "Avoid excess nitrogen; apply split doses.",
            "Drain fields periodically and avoid clipping seedlings.",
            "Remove infected stubble and weed hosts after harvest.",
        ],
    },
    "Brown spot": {
        "pathogen": "Bipolaris oryzae / Cochliobolus miyabeanus (fungus)",
        "symptoms": "Oval to circular brown spots with a grey or whitish centre and a "
                    "yellow halo, scattered over the leaf blade.",
        "management": [
            "Correct soil nutrient deficiency, especially potassium and silicon.",
            "Use certified, treated seed (fungicide seed treatment).",
            "Spray a recommended fungicide (e.g. propiconazole / mancozeb) when severe.",
            "Avoid water stress during crop growth.",
        ],
    },
    "Leaf smut": {
        "pathogen": "Entyloma oryzae (fungus)",
        "symptoms": "Small, slightly raised, angular black spots (sori) on both leaf "
                    "surfaces; leaf tips may dry out in heavy infection.",
        "management": [
            "Usually minor - maintain balanced fertilisation (avoid high nitrogen).",
            "Plough under / remove infected straw to reduce inoculum.",
            "Fungicide spray only if infection is widespread.",
        ],
    },
}
