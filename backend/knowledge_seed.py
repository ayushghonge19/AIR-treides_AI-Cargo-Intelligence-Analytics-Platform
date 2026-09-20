"""
AIR-treides — Knowledge Seed Data for pgvector RAG.
Contains realistic enterprise cargo handling guidelines, IATA regulatory SOPs,
and airline-specific cold chain/pharma instructions.
"""

import logging
from backend.vector_store import init_vector_table, ingest_guideline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEED_GUIDELINES = [
    {
        "title": "IATA DGR UN 3480 Lithium-Ion Battery Packaging Rules",
        "category": "Dangerous Goods",
        "airline": "ALL",
        "content": (
            "Under IATA Dangerous Goods Regulations (DGR) Packing Instruction 965, "
            "Lithium-ion cells and batteries (UN 3480) must be shipped at a State of Charge (SoC) "
            "not exceeding 30% of their rated capacity. They are strictly FORBIDDEN as cargo on passenger aircraft "
            "and must be labeled with the 'Cargo Aircraft Only' (CAO) label. Outer packaging must meet UN Performance "
            "Packaging Group II standards and carry the Class 9 Hazard Label and Lithium Battery Mark."
        ),
        "metadata": {"regulation": "IATA DGR", "section": "PI 965", "code": "UN3480"},
    },
    {
        "title": "IATA TCR Vaccine & Biopharma Temperature Control Standards",
        "category": "Pharma & Cold Chain",
        "airline": "ALL",
        "content": (
            "IATA Temperature Control Regulations (TCR) mandate strict range adherence for pharmaceuticals: "
            "CRT (Controlled Room Temperature): +15°C to +25°C; Refrigerated / Cold Chain: +2°C to +8°C; "
            "Frozen: -20°C (±5°C); Deep Frozen / Dry Ice: -80°C to -20°C. Temperature data loggers must be activated "
            "prior to loading and placed in the center of the thermal blanket or active ULD container (e.g. Envirotainer / CSafe)."
        ),
        "metadata": {"regulation": "IATA TCR", "temperature_ranges": ["+2 to +8C", "+15 to +25C", "-20C"]},
    },
    {
        "title": "Qatar Airways Cargo — QR Pharma SOP & Cold Chain Handling",
        "category": "Pharma & Cold Chain",
        "airline": "Qatar Airways",
        "content": (
            "Qatar Airways Cargo operates 'QR Pharma' via its Doha Hamad International Airport (HIA) hub. "
            "Features include automated temperature-controlled transit via refrigerated 'cool dollies' between aircraft "
            "and warehouse. QR Pharma offers two sub-products: 'QR Pharma Active' using motorized active containers, "
            "and 'QR Pharma Passive' using thermal blankets and dry ice. Ramp transfer time is capped at 45 minutes max."
        ),
        "metadata": {"airline": "Qatar Airways", "product": "QR Pharma", "hub": "DOH"},
    },
    {
        "title": "Emirates SkyCargo — Emirates Pharma & CoolChain Solutions",
        "category": "Pharma & Cold Chain",
        "airline": "Emirates",
        "content": (
            "Emirates SkyCargo provides GDP-certified (Good Distribution Practice) pharma handling at Dubai International (DXB) "
            "and Dubai World Central (DWC). The service includes Emirates Pharma Plus (+2°C to +8°C and +15°C to +25°C) and "
            "Emirates Pharma Active. Utilizes 50+ custom-built Cool Dollies with solar-powered refrigeration units during ramp transfer. "
            "Acceptance cutoff for temperature-sensitive cargo is 3 hours before departure."
        ),
        "metadata": {"airline": "Emirates", "product": "Emirates Pharma", "hub": "DXB"},
    },
    {
        "title": "IATA LAR Live Animals Regulations — Container & Ventilation Requirements",
        "category": "Live Animals",
        "airline": "ALL",
        "content": (
            "IATA Live Animals Regulations (LAR) requires all containers to meet Container Requirement (CR) standards. "
            "Containers must be escape-proof, ventilated on at least 3 sides (16% minimum total surface area for ventilation), "
            "and equipped with spacer bars to prevent airflow blockage when stowed adjacent to other cargo. Feeding and watering "
            "instructions must be affixed in English and clearly legible with duplicate copies in the airway bill (AWB) pouch."
        ),
        "metadata": {"regulation": "IATA LAR", "requirement": "CR 1-84"},
    },
    {
        "title": "Lufthansa Cargo — Fresh / Perishables & Seafood Handling",
        "category": "Perishables",
        "airline": "Lufthansa",
        "content": (
            "Lufthansa Cargo operates the Frankfurt Perishable Center (FPC), Europe's largest airport cool facility. "
            "Temperature zones are segmented into: +0°C to +2°C (Fresh Fish & Seafood), +2°C to +4°C (Meat & Dairy), "
            "+12°C to +15°C (Tropical Fruits & Flowers). Perishable shipments require mandatory advance electronic pre-declaration "
            "and priority offloading status on arrival."
        ),
        "metadata": {"airline": "Lufthansa", "facility": "FPC Frankfurt", "category": "Perishables"},
    },
    {
        "title": "Air Cargo Security — Known Consignor & Screening Standards",
        "category": "Security & Customs",
        "airline": "ALL",
        "content": (
            "Under ICAO Annex 17 and TSA / EU Aviation Security rules, all cargo transported on commercial passenger aircraft "
            "must undergo 100% piece-level security screening unless originated from a certified Known Consignor (KC). "
            "Primary screening methods include Dual-View X-Ray (XRY), Explosive Trace Detection (ETD), and Explosive Detection Dogs (EDD). "
            "Cargo that fails screening must be held in quarantine for a mandatory 24-hour cooling period."
        ),
        "metadata": {"regulation": "ICAO Annex 17", "screening": ["XRY", "ETD", "EDD"]},
    },
]


def seed_knowledge_base():
    """Initializes table and embeds seed documents."""
    logger.info("Initializing vector table in Supabase...")
    init_vector_table()
    
    logger.info(f"Embedding and ingesting {len(SEED_GUIDELINES)} seed guidelines...")
    for item in SEED_GUIDELINES:
        doc_id = ingest_guideline(
            title=item["title"],
            category=item["category"],
            content=item["content"],
            airline=item["airline"],
            metadata=item.get("metadata", {}),
        )
        logger.info(f"Ingested [{doc_id}]: {item['title']}")
    
    logger.info("Knowledge base seeding complete!")


if __name__ == "__main__":
    seed_knowledge_base()
