"""Task-specific configuration for Humathèque thesis metadata extraction."""

from __future__ import annotations

import json

DEFAULT_SOURCE_DATASET = "Geraldine/humatheque-vlm-sudoc-grounded"
DEFAULT_GROUND_TRUTH_COLUMN = "sudoc_record_templated"
DEFAULT_IMAGE_COLUMN = "image_uri"

THESIS_DEGREE_TYPE_VALUES = [
    "Thèse d'État",
    "Thèse de doctorat",
    "Thèse de 3e cycle",
    "Thèse d'université",
    "Thèse de docteur-ingénieur",
    "Thèse d'exercice",
]

DISSERTATION_DEGREE_TYPE_VALUES = [
    "Habilitation à diriger des recherches",
    "Mémoire de DEA",
    "Mémoire de DES",
    "Mémoire de DESS",
    "Mémoire de DU",
    "Mémoire de DIU",
    "Mémoire de DUT",
    "Mémoire de maîtrise",
    "Mémoire de master professionnel 1re année",
    "Mémoire de master professionnel 2e année",
    "Mémoire de master recherche 1re année",
    "Mémoire de master recherche 2e année",
]

OAI_DISCIPLINE_VALUES = [
    "Informatique, information, généralités",
    "Informatique",
    "Bibliothéconomie et sciences de l'information",
    "Organisations générales et muséologie",
    "Médias d'information, journalisme, édition",
    "Manuscrits et livres rares",
    "Philosophie, psychologie",
    "Métaphysique",
    "Epistémologie, causalité, genre humain",
    "Phénomènes paranormaux, pseudosciences",
    "Les divers systèmes et écoles philosophiques",
    "Psychologie",
    "Logique",
    "Morale (éthique)",
    "Philosophie de l'Antiquité, du Moyen Âge, de l'Orient",
    "Philosophie occidentale moderne et philosophies non orientales",
    "Religion",
    "Philosophie et théorie de la religion",
    "Bible",
    "Théologie chrétienne",
    "Théologie morale et pratiques chrétiennes",
    "Eglises locales, ordres religieux chrétiens",
    "Théologie chrétienne et société, ecclésiologie",
    "Histoire et géographie du christianisme et de l'Eglise chrétienne",
    "Confessions et sectes de l'Eglise chrétienne",
    "Autres religions",
    "Sciences sociales, sociologie, anthropologie",
    "Statistiques générales",
    "Science politique",
    "Economie",
    "Droit",
    "Administration publique. Arts et science militaires",
    "Problèmes et services sociaux",
    "Education et enseignement",
    "Commerce, communications, transports",
    "Ethnologie",
    "Langues et linguistique",
    "Linguistique générale",
    "Langue anglaise. Anglo-saxon",
    "Langues germaniques. Allemand",
    "Langues romanes. Français",
    "Langues italienne, roumaine, rhéto-romane",
    "Langues espagnole et portugaise",
    "Langues italiques. Latin",
    "Langues helléniques. Grec classique",
    "Autres langues",
    "Sciences de la nature et mathématiques",
    "Mathématiques",
    "Astronomie, cartographie, géodésie",
    "Physique",
    "Chimie, minéralogie, cristallographie",
    "Sciences de la terre",
    "Paléontologie. Paléozoologie",
    "Sciences de la vie, biologie, biochimie",
    "Plantes. Botanique",
    "Animaux. Zoologie",
    "Technologie (Sciences appliquées)",
    "Médecine et santé",
    "Sciences de l'ingénieur",
    "Agronomie, agriculture et médecine vétérinaire",
    "Economie domestique. Vie familiale",
    "Gestion et organisation de l'entreprise",
    "Génie chimique, technologies alimentaires",
    "Fabrication industrielle",
    "Fabrication de produits à usages spécifiques",
    "Bâtiments",
    "Arts. Beaux-arts et arts décoratifs",
    "Urbanisme",
    "Architecture",
    "Arts plastiques. Sculpture",
    "Dessin. Arts décoratifs",
    "Peinture",
    "Arts graphiques",
    "Photographie et les photographies, art numérique",
    "Musique",
    "Arts du spectacle, loisirs",
    "Sport",
    "Histoire et critique littéraires, rhétorique",
    "Littérature américaine en anglais",
    "Littératures anglaise et anglo-saxonne",
    "Littérature allemande",
    "Littérature de langues romanes. Littérature française",
    "Littérature italienne",
    "Littératures espagnole et portugaise",
    "Littérature latine",
    "Littérature grecque",
    "Littératures des autres langues",
    "Géographie et histoire",
    "Géographie et voyages",
    "Biographies générales, généalogie, emblèmes",
    "Histoire ancienne et préhistoire",
    "Histoire moderne et contemporaine de l'Europe",
    "Histoire générale de la France",
    "Histoire générale de l'Asie, Orient, Extrême-Orient",
    "Histoire générale de l'Afrique",
    "Histoire générale de l'Amérique du Nord",
    "Histoire générale de l'Amérique du Sud",
    "Histoire générale des autres parties du monde, des mondes extraterrestres. Iles du Pacifique",
]


def build_eval_prompt(doc_type: str) -> str:
    """Build extraction prompt dynamically from document type."""
    doc_type_norm = str(doc_type or "").strip().lower()
    if doc_type_norm == "memoire":
        degree_type_values = DISSERTATION_DEGREE_TYPE_VALUES
        document_label = "graduate dissertation"
    else:
        degree_type_values = THESIS_DEGREE_TYPE_VALUES
        document_label = "graduate thesis"

    return f"""Extract the document title from this {document_label} cover page.
Output ONLY valid JSON:
{{
  "title": "Main title as it appears on the title page",
  "subtitle": "Subtitle or remainder of the title, usually following a colon; null if not present",
  "author": "Full name of the author (student) who wrote the {document_label}",
  "degree_type": "Academic degree sought by the author. Possible values are {json.dumps(degree_type_values, ensure_ascii=False)}",
  "discipline": "Academic field or discipline of the {document_label}.",
  "granting_institution": "Institution where the {document_label} was submitted and the degree is granted",
  "co_tutelle_institutions": "List of institutions involved in a joint supervision or co-tutelle agreement; empty list if none",
  "doctoral_school": "Doctoral school or graduate program, if explicitly mentioned",
  "defense_year": "Year the {document_label} was defended. Format yyyy",
  "advisor": "Main {document_label} advisor(s) or supervisor(s). Use | as separator.",
  "jury_president": "President/chair of the jury or examination committee. Do not include this person in committee_members.",
  "reviewers": "Official reviewers/rapporteurs. Use | as separator. Do not include these people in committee_members.",
  "committee_members": "All jury/examination committee members except president/chair and reviewers/rapporteurs. This field MUST include advisor(s)/supervisor(s), duplicating them from advisor when present. Use | as separator.",
  "language": "Language in ISO 639-3 codes. Example: fre, eng, ita...",
  "confidence": "Confidence score between 0.0 and 1.0 indicating reliability of the extracted metadata"
}}"""


def build_default_task_prompt() -> str:
    """Default prompt for runs when doc_type is unknown."""
    return build_eval_prompt("these")
