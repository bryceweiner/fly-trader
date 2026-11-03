"""Population index sets for the FlyWire FAFB v783 connectome.

The neuron table is flybrain's ``neurons.csv.gz`` (139,255 canonical root_ids, the same set and
order the connectome edges refer to) left-joined on ``root_id`` with the Schlegel et al. 2024
annotation TSV (``Supplemental_file1_neuron_annotations.tsv``: cell_class / cell_sub_class /
cell_type / top_nt). Neurotransmitter = ``top_nt`` when annotated, else flybrain's ``nt_type``
mapped to the same vocabulary.

Populations are *contiguous index ranges*: neurons are sorted by (population_id, root_id) so a
population is a slice, never a gather, on the GPU. Membership is first-match in
``POPULATION_ORDER``; everything unmatched is ``OTHER``.

Valence tables (config/mbon_valence.json): Aso et al. 2014 eLife 3:e04580 (aversion-driving MBONs
are glutamatergic, attraction-driving MBONs GABAergic/cholinergic) and 3:e04577 (PAM = reward,
PPL1 = punishment). KC sparse code: Dasgupta, Stevens, Navlakha 2017 Science.

flybrain's 63 behavioural groups (``determine_group`` from snedea/flybrain
``scripts/build_connectome.py``) are recomputed verbatim so group rates stay comparable with the
JavaScript demo; the DRIVE_HUNGER group (46 pars-intercerebralis neurons) is our HUNGER population.

Fallback (``config.CELL_TYPES_SOURCE == 'flybrain'`` or the TSV is absent): flybrain's
``classification.csv.gz`` supplies the classes; MBON valence is assigned by neurotransmitter
(glutamate -> avoid, acetylcholine/gaba -> approach) and DAN valence by synapse counts onto avoid
vs approach MBONs (reward if more onto avoid MBONs: reward DANs depress avoidance synapses);
glomeruli come from the Codex ``consolidated_cell_types.csv.gz`` if it types >= 30 ORN
glomeruli, else a seeded random partition into 50 pseudo-glomeruli.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config

RAW_DIR = config.BRAIN_DIR / "raw"
NEURONS_CSV = RAW_DIR / "neurons.csv.gz"
CLASSIFICATION_CSV = RAW_DIR / "classification.csv.gz"
ANNOTATIONS_TSV = RAW_DIR / "Supplemental_file1_neuron_annotations.tsv"
CONSOLIDATED_TYPES_CSV = RAW_DIR / "consolidated_cell_types.csv.gz"
MBON_VALENCE_JSON = config.REPO_ROOT / "config" / "mbon_valence.json"

EXPECTED_N = 139_255
EXPECTED_KC = 5_177
EXPECTED_MBON = 96

# Local constants (not in config.py): pseudo-glomerulus fallback.
MIN_TYPED_GLOMERULI = 30
PSEUDO_GLOMERULI = 50

# First match wins; the remainder is OTHER.
POPULATION_ORDER: list[str] = [
    "KC", "MBON_APP", "MBON_AV", "MBON_OTHER",
    "DAN_PAM", "DAN_PPL1", "DAN_OTHER",
    "ORN_DANGER", "ORN_FOOD", "ALPN", "ALLN", "LH",
    "GRN_SWEET", "GRN_BITTER", "GRN_OTHER",
    "MECH_JO", "MECH_BRISTLE", "MECH_OTHER",
    "THERMO_WARM", "THERMO_COOL", "THERMO_OTHER",
    "CX", "HUNGER", "DESCENDING", "VISUAL", "OTHER",
]

# kind for the ``populations`` table: sensory | central | mb | motor | other
POPULATION_KIND: dict[str, str] = {
    "KC": "mb", "MBON_APP": "mb", "MBON_AV": "mb", "MBON_OTHER": "mb",
    "DAN_PAM": "mb", "DAN_PPL1": "mb", "DAN_OTHER": "mb",
    "ORN_DANGER": "sensory", "ORN_FOOD": "sensory", "ALPN": "central", "ALLN": "central", "LH": "central",
    "GRN_SWEET": "sensory", "GRN_BITTER": "sensory", "GRN_OTHER": "sensory",
    "MECH_JO": "sensory", "MECH_BRISTLE": "sensory", "MECH_OTHER": "sensory",
    "THERMO_WARM": "sensory", "THERMO_COOL": "sensory", "THERMO_OTHER": "sensory",
    "CX": "central", "HUNGER": "central", "DESCENDING": "motor", "VISUAL": "sensory", "OTHER": "other",
}

NT_FLYBRAIN_TO_NAME: dict[str, str] = {
    "ACH": "acetylcholine", "GABA": "gaba", "GLUT": "glutamate", "DA": "dopamine",
    "SER": "serotonin", "OA": "octopamine", "OCT": "octopamine",
}

# ---------------------------------------------------------------------------------------------
# flybrain groups, ported verbatim from snedea/flybrain scripts/build_connectome.py
# ---------------------------------------------------------------------------------------------
FLYBRAIN_GROUPS: list[tuple[str, str]] = [
    ("VIS_R1R6", "sensory"), ("VIS_R7R8", "sensory"), ("VIS_ME", "sensory"), ("VIS_LO", "sensory"),
    ("VIS_LC", "sensory"), ("VIS_LPTC", "sensory"), ("OLF_ORN_FOOD", "sensory"), ("OLF_ORN_DANGER", "sensory"),
    ("OLF_LN", "sensory"), ("OLF_PN", "sensory"), ("MECH_BRISTLE", "sensory"), ("MECH_JO", "sensory"),
    ("MECH_CHORD", "sensory"), ("ANTENNAL_MECH", "sensory"), ("THERMO_WARM", "sensory"), ("THERMO_COOL", "sensory"),
    ("NOCI", "sensory"),
    ("MB_KC", "central"), ("MB_APL", "central"), ("MB_MBON_APP", "central"), ("MB_MBON_AV", "central"),
    ("MB_DAN_REW", "central"), ("MB_DAN_PUN", "central"), ("LH_APP", "central"), ("LH_AV", "central"),
    ("CX_EPG", "central"), ("CX_PFN", "central"), ("CX_FC", "central"), ("CX_HDELTA", "central"),
    ("SEZ_FEED", "central"), ("SEZ_GROOM", "central"), ("SEZ_WATER", "central"), ("GUS_GRN_SWEET", "central"),
    ("GUS_GRN_BITTER", "central"), ("GUS_GRN_WATER", "central"), ("GNG_DESC", "central"), ("CLOCK_DN", "central"),
    ("DRIVE_HUNGER", "drives"), ("DRIVE_FEAR", "drives"), ("DRIVE_FATIGUE", "drives"), ("DRIVE_CURIOSITY", "drives"),
    ("DRIVE_GROOM", "drives"),
    ("DN_WALK", "motor"), ("DN_FLIGHT", "motor"), ("DN_TURN", "motor"), ("DN_BACKUP", "motor"),
    ("DN_STARTLE", "motor"), ("VNC_CPG", "motor"), ("MN_LEG_L1", "motor"), ("MN_LEG_R1", "motor"),
    ("MN_LEG_L2", "motor"), ("MN_LEG_R2", "motor"), ("MN_LEG_L3", "motor"), ("MN_LEG_R3", "motor"),
    ("MN_WING_L", "motor"), ("MN_WING_R", "motor"), ("MN_PROBOSCIS", "motor"), ("MN_HEAD", "motor"),
    ("MN_ABDOMEN", "motor"),
    ("GENERIC_SENSORY", "sensory"), ("GENERIC_CENTRAL", "central"), ("GENERIC_DRIVES", "drives"),
    ("GENERIC_MOTOR", "motor"),
]
FLYBRAIN_GROUP_NAMES: list[str] = [name for name, _ in FLYBRAIN_GROUPS]
FLYBRAIN_GROUP_ID: dict[str, int] = {name: i for i, name in enumerate(FLYBRAIN_GROUP_NAMES)}
assert len(FLYBRAIN_GROUPS) == 63


def determine_group(flow: str, super_class: str, cls: str, sub_class: str, region: str, side: str = "") -> str:
    """Map classification fields to one of the 63 group names (59 named + 4 generic).

    Verbatim port of flybrain ``scripts/build_connectome.py::determine_group`` (inputs are the
    lower-cased flow / super_class / class / sub_class / side columns of classification.csv).
    """
    # --- Visual system ---
    if "visual" in cls or "optic" in cls or "visual" in super_class or "optic" in super_class:
        if "photo_receptor" in sub_class or "photo" in sub_class:
            if "r7" in sub_class or "r8" in sub_class or "uv" in sub_class or "pale" in sub_class:
                return "VIS_R7R8"
            return "VIS_R1R6"
        if "ocellar" in sub_class or "ocellar" in cls:
            return "VIS_R1R6"
        if "lptc" in sub_class or "tangential" in sub_class:
            return "VIS_LPTC"
        if "lc" in sub_class or "loom" in sub_class or "lobula_columnar" in sub_class:
            return "VIS_LC"
        if "lobula" in sub_class and "plate" in sub_class:
            return "VIS_LPTC"
        if "lobula" in sub_class or sub_class == "lo":
            return "VIS_LO"
        if "medulla" in sub_class or "tm" in sub_class or "mi" in sub_class:
            return "VIS_ME"
        return "VIS_ME"

    # --- Olfactory system ---
    if "olfact" in cls:
        if "pheromone" in sub_class or "avers" in sub_class or "danger" in sub_class:
            return "OLF_ORN_DANGER"
        return "OLF_ORN_FOOD"
    if cls == "alpn":
        return "OLF_PN"
    if cls == "alln":
        return "OLF_LN"

    # --- Gustatory system ---
    if "gustat" in cls:
        if "bitter" in sub_class:
            return "GUS_GRN_BITTER"
        if "water" in sub_class:
            return "GUS_GRN_WATER"
        return "GUS_GRN_SWEET"

    # --- Mechanosensory system ---
    if "mechano" in cls:
        if "wind" in sub_class or "gravity" in sub_class or "auditory" in sub_class:
            return "MECH_JO"
        if "groom" in sub_class:
            return "MECH_BRISTLE"
        if "bristle" in sub_class or "taste_peg" in sub_class:
            return "MECH_BRISTLE"
        if "chord" in sub_class or "propriocep" in sub_class:
            return "MECH_CHORD"
        if "antenna" in sub_class:
            return "ANTENNAL_MECH"
        return "MECH_BRISTLE"

    # --- Thermosensory ---
    if "thermo" in cls:
        if "cool" in sub_class or "cold" in sub_class:
            return "THERMO_COOL"
        return "THERMO_WARM"

    # --- Hygrosensory -> thermosensory ---
    if "hygro" in cls:
        if "dry" in sub_class:
            return "THERMO_WARM"
        return "THERMO_COOL"

    # --- Nociceptive ---
    if "nocicep" in cls:
        return "NOCI"

    # --- Unknown sensory -> generic sensory ---
    if "unknown_sensory" in cls:
        return "MECH_BRISTLE"

    # --- Mushroom body: Kenyon cells ---
    if "kenyon" in cls or cls == "kc":
        return "MB_KC"

    # --- Mushroom body output neurons (MBONs) ---
    if cls == "mbon":
        return "MB_MBON_APP"

    # --- Mushroom body input neurons (MBINs) ---
    if cls == "mbin":
        return "MB_DAN_REW"

    # --- Dopaminergic neurons (DANs) ---
    if cls == "dan":
        return "MB_DAN_REW"

    # --- Lateral horn ---
    if cls == "lhln" or cls == "lhcent":
        return "LH_APP"

    # --- Central complex ---
    if cls == "cx":
        if "ring" in sub_class:
            return "CX_EPG"
        if "tangential" in sub_class:
            return "CX_HDELTA"
        if "columnar" in sub_class:
            return "CX_PFN"
        return "CX_FC"

    # --- Antennal lobe interneurons ---
    if cls == "alin":
        return "OLF_LN"
    if cls == "alon":
        return "OLF_PN"

    # --- TuBu neurons (tubercle to bulb, visual -> CX pathway) ---
    if cls == "tubu":
        return "CX_EPG"

    # --- TPN (taste projection neurons) ---
    if cls == "tpn":
        if "water" in sub_class:
            return "GUS_GRN_WATER"
        return "GUS_GRN_SWEET"

    # --- MAL (medial accessory lobe) neurons ---
    if cls == "mal":
        return "CX_FC"

    # --- Ascending neurons ---
    if "ascending" in super_class or cls == "an":
        return "GNG_DESC"

    # --- Descending neurons ---
    if "descend" in super_class or "descend" in cls:
        if "dn1p" in sub_class:
            return "DN_STARTLE"
        if "dn3" in sub_class:
            return "DN_WALK"
        if "walk" in sub_class or "locomot" in sub_class:
            return "DN_WALK"
        if "flight" in sub_class:
            return "DN_FLIGHT"
        if "turn" in sub_class:
            return "DN_TURN"
        if "back" in sub_class:
            return "DN_BACKUP"
        if "startle" in sub_class or "escape" in sub_class or "giant" in sub_class:
            return "DN_STARTLE"
        return "GNG_DESC"

    # --- Motor neurons ---
    if "motor" in cls or "motor" in super_class:
        if "proboscis" in sub_class:
            return "MN_PROBOSCIS"
        if "neck" in sub_class or "head" in sub_class:
            return "MN_HEAD"
        if "abdom" in sub_class or "crop" in sub_class:
            return "MN_ABDOMEN"
        if "ingestion" in sub_class or "haustellum" in sub_class or "salivary" in sub_class:
            return "SEZ_FEED"
        if "eye" in sub_class or "antenna" in sub_class:
            return "MN_HEAD"
        return "VNC_CPG"

    # --- Clock neurons ---
    if "clock" in cls or "circadian" in cls:
        return "CLOCK_DN"

    # --- Endocrine / neurosecretory ---
    if "endocrine" in super_class:
        if "pars_lateralis" in cls:
            return "DRIVE_FATIGUE"
        return "DRIVE_HUNGER"

    # --- Fallback: use flow to pick a region-appropriate generic ---
    if flow == "afferent":
        return "GENERIC_SENSORY"
    if flow == "efferent":
        return "GENERIC_MOTOR"
    return "GENERIC_CENTRAL"


# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------
def load_neuron_table(annotations: str | Path | None = None, source: str | None = None) -> tuple[pd.DataFrame, str]:
    """flybrain neurons (canonical order) + classification + annotations (left join).

    Returns (table, source) where source is 'annotations' or 'flybrain'. Columns:
    root_id (int64), fb_nt, fb_flow, fb_super_class, fb_class, fb_sub_class, fb_side (lower-cased,
    as flybrain sees them), flybrain_group (uint16), annotated (bool), super_class, cell_class,
    cell_sub_class, cell_type, top_nt, nt.
    """
    source = source or config.CELL_TYPES_SOURCE
    neu = pd.read_csv(NEURONS_CSV, usecols=["root_id", "nt_type"], dtype={"root_id": np.int64, "nt_type": str},
                      keep_default_na=False)
    if len(neu) != EXPECTED_N:
        print(f"[populations] WARNING neurons.csv has {len(neu)} rows, expected {EXPECTED_N}", file=sys.stderr)
    cls = pd.read_csv(CLASSIFICATION_CSV, dtype=str, keep_default_na=False)
    cls["root_id"] = cls["root_id"].astype(np.int64)
    for c in ("flow", "super_class", "class", "sub_class", "side"):
        cls[c] = cls[c].str.strip().str.lower()
    cls = cls.rename(columns={"flow": "fb_flow", "super_class": "fb_super_class", "class": "fb_class",
                              "sub_class": "fb_sub_class", "side": "fb_side"})
    cls = cls[["root_id", "fb_flow", "fb_super_class", "fb_class", "fb_sub_class", "fb_side"]]
    df = neu.rename(columns={"nt_type": "fb_nt"}).merge(cls, on="root_id", how="left")
    for c in ("fb_flow", "fb_super_class", "fb_class", "fb_sub_class", "fb_side"):
        df[c] = df[c].fillna("")
    df["fb_nt"] = df["fb_nt"].str.strip().str.upper()

    # flybrain group labels (classification.csv rows; neurons without a row keep GENERIC_CENTRAL)
    groups = np.full(len(df), FLYBRAIN_GROUP_ID["GENERIC_CENTRAL"], dtype=np.uint16)
    keyed = df[["fb_flow", "fb_super_class", "fb_class", "fb_sub_class", "fb_side"]].drop_duplicates()
    cache: dict[tuple, int] = {}
    for row in keyed.itertuples(index=False):
        cache[tuple(row)] = FLYBRAIN_GROUP_ID[determine_group(row[0], row[1], row[2], row[3], "central", row[4])]
    keys = list(zip(df["fb_flow"], df["fb_super_class"], df["fb_class"], df["fb_sub_class"], df["fb_side"]))
    has_row = df["root_id"].isin(cls["root_id"]).to_numpy()
    groups[has_row] = np.array([cache[k] for k, h in zip(keys, has_row) if h], dtype=np.uint16)
    df["flybrain_group"] = groups

    ann_path = Path(annotations) if annotations else ANNOTATIONS_TSV
    use_ann = source == "annotations" and ann_path.exists()
    if source == "annotations" and not ann_path.exists():
        print(f"[populations] WARNING annotations TSV missing at {ann_path}; falling back to flybrain "
              f"classification", file=sys.stderr)
    if use_ann:
        ann = pd.read_csv(ann_path, sep="\t", dtype=str, keep_default_na=False, low_memory=False,
                          usecols=["root_id", "super_class", "cell_class", "cell_sub_class", "cell_type", "top_nt"])
        ann["root_id"] = ann["root_id"].astype(np.int64)
        ann = ann.drop_duplicates("root_id")
        ann["annotated"] = True
        df = df.merge(ann, on="root_id", how="left")
        df["annotated"] = df["annotated"].fillna(False).astype(bool)
        for c in ("super_class", "cell_class", "cell_sub_class", "cell_type", "top_nt"):
            df[c] = df[c].fillna("").str.strip()
        src = "annotations"
    else:
        df["annotated"] = False
        for c in ("super_class", "cell_class", "cell_sub_class", "cell_type", "top_nt"):
            df[c] = ""
        src = "flybrain"
    fb_nt_named = df["fb_nt"].map(NT_FLYBRAIN_TO_NAME).fillna("")
    df["nt"] = np.where(df["top_nt"] != "", df["top_nt"].str.lower(), fb_nt_named)
    return df, src


def load_mbon_valence(path: str | Path | None = None) -> dict:
    with open(path or MBON_VALENCE_JSON) as f:
        return json.load(f)


def mbon_base_type(cell_type: str) -> str:
    """Exact type only: '-like' annotations and combined labels ('MBON25,MBON34') stay unassignable
    (no valence evidence for '-like' cells; literature review 2026-09-12)."""
    return cell_type.strip()


# ---------------------------------------------------------------------------------------------
# Populations
# ---------------------------------------------------------------------------------------------
@dataclass
class Populations:
    order: list[str]
    ranges: dict[str, tuple[int, int]]
    root_ids: np.ndarray                 # int64 [N], population order
    nt: np.ndarray                       # str [N]
    flybrain_group: np.ndarray           # uint16 [N]
    flybrain_group_names: list[str]
    glomerulus_of_orn: np.ndarray        # int32 [n_ORN_FOOD], -1 unknown
    glomerulus_names: list[str]
    canonical_index: np.ndarray          # int64 [N]: sorted position i holds flybrain row canonical_index[i]
    source: str = "annotations"
    cell_type: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=object))
    kind: dict[str, str] = field(default_factory=lambda: dict(POPULATION_KIND))

    @property
    def N(self) -> int:
        return int(len(self.root_ids))

    def n(self, name: str) -> int:
        a, b = self.ranges[name]
        return b - a

    def index(self, name: str) -> np.ndarray:
        a, b = self.ranges[name]
        return np.arange(a, b, dtype=np.int64)

    def slice(self, name: str) -> slice:
        a, b = self.ranges[name]
        return slice(a, b)

    def population_id(self) -> np.ndarray:
        pid = np.empty(self.N, dtype=np.int16)
        for i, name in enumerate(self.order):
            a, b = self.ranges[name]
            pid[a:b] = i
        return pid

    def counts(self) -> dict[str, int]:
        return {name: self.n(name) for name in self.order}

    def inverse_index(self) -> np.ndarray:
        """canonical flybrain row -> sorted position."""
        inv = np.empty(self.N, dtype=np.int64)
        inv[self.canonical_index] = np.arange(self.N, dtype=np.int64)
        return inv


def _assign_annotations(df: pd.DataFrame, valence: dict) -> np.ndarray:
    cc = df["cell_class"].to_numpy(dtype=object)
    sc = df["cell_sub_class"].to_numpy(dtype=object)
    ct = df["cell_type"].to_numpy(dtype=object)
    base = np.array([mbon_base_type(t) for t in ct], dtype=object)
    sup = np.where(df["annotated"].to_numpy(), df["super_class"].to_numpy(dtype=object),
                   df["fb_super_class"].to_numpy(dtype=object))
    approach = set(valence["approach"])
    avoid = set(valence["avoid"])
    rew = valence.get("dan_reward_prefix", "PAM")
    pun = valence.get("dan_punishment_prefix", "PPL1")
    is_mbon = cc == "MBON"
    is_dan = cc == "DAN"
    is_olf = cc == "olfactory"
    is_gus = cc == "gustatory"
    is_mech = cc == "mechanosensory"
    is_thermo = cc == "thermosensory"
    is_hygro = cc == "hygrosensory"
    starts = np.char.startswith
    ct_s = ct.astype(str)
    rules = {
        "KC": cc == "Kenyon_Cell",
        "MBON_APP": is_mbon & np.isin(base, list(approach)),
        "MBON_AV": is_mbon & np.isin(base, list(avoid)),
        "MBON_OTHER": is_mbon,
        "DAN_PAM": is_dan & starts(ct_s, rew),
        "DAN_PPL1": is_dan & starts(ct_s, pun),
        "DAN_OTHER": is_dan,
        "ORN_DANGER": is_olf & (sc == "pheromone"),
        "ORN_FOOD": is_olf,
        "ALPN": cc == "ALPN",
        "ALLN": cc == "ALLN",
        "LH": np.isin(cc, ["LHLN", "LHCENT"]),
        "GRN_SWEET": is_gus & (sc == "sugar/water"),
        "GRN_BITTER": is_gus & (sc == "bitter"),
        "GRN_OTHER": is_gus,
        "MECH_JO": is_mech & np.isin(sc, ["wind_gravity", "auditory"]),
        "MECH_BRISTLE": is_mech & np.isin(sc, ["eye bristle", "head bristle"]),
        "MECH_OTHER": is_mech,
        "THERMO_WARM": (is_thermo & (sc == "heating")) | (is_hygro & (sc == "dry")),
        "THERMO_COOL": (is_thermo & (sc == "cold")) | (is_hygro & np.isin(sc, ["cooling", "moist", "evaporative_cooling"])),
        "THERMO_OTHER": is_thermo | is_hygro,
        "CX": cc == "CX",
        "HUNGER": df["flybrain_group"].to_numpy() == FLYBRAIN_GROUP_ID["DRIVE_HUNGER"],
        "DESCENDING": sup == "descending",
        "VISUAL": np.isin(sup, ["optic", "visual_projection", "visual_centrifugal"]) | (cc == "visual"),
    }
    return _first_match(rules, len(df))


def _contains(arr: np.ndarray, needle: str) -> np.ndarray:
    return np.array([needle in s for s in arr], dtype=bool)


def _assign_flybrain(df: pd.DataFrame, dan_valence: np.ndarray | None) -> np.ndarray:
    """Fallback partition from flybrain classification.csv (lower-cased fields)."""
    cls = df["fb_class"].to_numpy(dtype=object)
    sub = df["fb_sub_class"].to_numpy(dtype=object)
    sup = df["fb_super_class"].to_numpy(dtype=object)
    nt = df["nt"].to_numpy(dtype=object)
    is_kc = _contains(cls, "kenyon") | (cls == "kc")
    is_mbon = cls == "mbon"
    is_dan = cls == "dan"
    is_olf = _contains(cls, "olfact")
    is_gus = _contains(cls, "gustat")
    is_mech = _contains(cls, "mechano")
    is_thermo = _contains(cls, "thermo")
    is_hygro = _contains(cls, "hygro")
    if dan_valence is None:
        dan_valence = np.zeros(len(df), dtype=np.int8)
    rules = {
        "KC": is_kc,
        "MBON_APP": is_mbon & np.isin(nt, ["acetylcholine", "gaba"]),
        "MBON_AV": is_mbon & (nt == "glutamate"),
        "MBON_OTHER": is_mbon,
        "DAN_PAM": is_dan & (dan_valence > 0),
        "DAN_PPL1": is_dan & (dan_valence < 0),
        "DAN_OTHER": is_dan,
        "ORN_DANGER": is_olf & (_contains(sub, "pheromone") | _contains(sub, "avers") | _contains(sub, "danger")),
        "ORN_FOOD": is_olf,
        "ALPN": cls == "alpn",
        "ALLN": cls == "alln",
        "LH": (cls == "lhln") | (cls == "lhcent"),
        "GRN_SWEET": is_gus & (_contains(sub, "sugar") | _contains(sub, "water")),
        "GRN_BITTER": is_gus & _contains(sub, "bitter"),
        "GRN_OTHER": is_gus,
        "MECH_JO": is_mech & (_contains(sub, "wind") | _contains(sub, "gravity") | _contains(sub, "auditory")),
        "MECH_BRISTLE": is_mech & _contains(sub, "bristle"),
        "MECH_OTHER": is_mech,
        "THERMO_WARM": (is_thermo & (_contains(sub, "heat") | _contains(sub, "warm"))) | (is_hygro & _contains(sub, "dry")),
        "THERMO_COOL": (is_thermo & (_contains(sub, "cold") | _contains(sub, "cool")))
                       | (is_hygro & (_contains(sub, "cool") | _contains(sub, "moist") | _contains(sub, "evap"))),
        "THERMO_OTHER": is_thermo | is_hygro,
        "CX": cls == "cx",
        "HUNGER": df["flybrain_group"].to_numpy() == FLYBRAIN_GROUP_ID["DRIVE_HUNGER"],
        "DESCENDING": _contains(sup, "descend"),
        "VISUAL": np.isin(sup, ["optic", "visual_projection", "visual_centrifugal"]) | (cls == "visual"),
    }
    return _first_match(rules, len(df))


def _first_match(rules: dict[str, np.ndarray], n: int) -> np.ndarray:
    pid = np.full(n, -1, dtype=np.int16)
    for name in POPULATION_ORDER:
        if name == "OTHER":
            continue
        m = rules[name] & (pid < 0)
        pid[m] = POPULATION_ORDER.index(name)
    pid[pid < 0] = POPULATION_ORDER.index("OTHER")
    return pid


def fallback_dan_valence(df: pd.DataFrame, pid_partial: np.ndarray, edges: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """+1 reward / -1 punishment / 0 unknown per canonical neuron, from synapse counts onto avoid vs
    approach MBONs (reward DANs innervate the compartments whose MBONs drive avoidance)."""
    pre, post, syn = edges
    app = POPULATION_ORDER.index("MBON_APP")
    av = POPULATION_ORDER.index("MBON_AV")
    n = len(df)
    onto_app = np.bincount(pre, weights=syn * (pid_partial[post] == app), minlength=n)
    onto_av = np.bincount(pre, weights=syn * (pid_partial[post] == av), minlength=n)
    val = np.zeros(n, dtype=np.int8)
    val[onto_av > onto_app] = 1
    val[(onto_app > 0) & (onto_app >= onto_av)] = -1
    return val


def _glomeruli_annotations(cell_types: np.ndarray) -> tuple[np.ndarray, list[str]]:
    names: list[str] = sorted({t[len("ORN_"):] for t in cell_types if t.startswith("ORN_") and len(t) > 4})
    lut = {n: i for i, n in enumerate(names)}
    gid = np.array([lut.get(t[len("ORN_"):], -1) if t.startswith("ORN_") else -1 for t in cell_types], dtype=np.int32)
    return gid, names


def _glomeruli_fallback(root_ids: np.ndarray, seed: int) -> tuple[np.ndarray, list[str]]:
    if CONSOLIDATED_TYPES_CSV.exists():
        ct = pd.read_csv(CONSOLIDATED_TYPES_CSV, dtype=str, keep_default_na=False, usecols=["root_id", "primary_type"])
        ct["root_id"] = ct["root_id"].astype(np.int64)
        lut = dict(zip(ct["root_id"].to_numpy(), ct["primary_type"].to_numpy()))
        types = np.array([lut.get(r, "") for r in root_ids], dtype=object)
        gid, names = _glomeruli_annotations(types)
        if len(names) >= MIN_TYPED_GLOMERULI:
            return gid, names
    rng = np.random.default_rng(seed)
    gid = rng.integers(0, PSEUDO_GLOMERULI, size=len(root_ids)).astype(np.int32)
    return gid, [f"pseudo_{i:02d}" for i in range(PSEUDO_GLOMERULI)]


def build_populations(annotations: str | Path | None = None, source: str | None = None,
                      edges: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
                      valence_path: str | Path | None = None, seed: int = 0, verbose: bool = True) -> Populations:
    """Partition the neuron table into contiguous populations.

    ``edges`` = (pre_canonical, post_canonical, syn_count) arrays; only needed for the flybrain
    fallback (DAN valence by synapse counts).
    """
    df, src = load_neuron_table(annotations, source)
    valence = load_mbon_valence(valence_path)
    if src == "annotations":
        pid = _assign_annotations(df, valence)
    else:
        pid = _assign_flybrain(df, None)
        if edges is not None:
            pid = _assign_flybrain(df, fallback_dan_valence(df, pid, edges))
        elif verbose:
            print("[populations] WARNING flybrain fallback without edges: DANs are all DAN_OTHER", file=sys.stderr)
    root_ids = df["root_id"].to_numpy(dtype=np.int64)
    perm = np.lexsort((root_ids, pid))  # by population, then root_id
    pid_sorted = pid[perm]
    ranges: dict[str, tuple[int, int]] = {}
    for i, name in enumerate(POPULATION_ORDER):
        a = int(np.searchsorted(pid_sorted, i, side="left"))
        b = int(np.searchsorted(pid_sorted, i, side="right"))
        ranges[name] = (a, b)
    cell_type = df["cell_type"].to_numpy(dtype=object)[perm]
    a, b = ranges["ORN_FOOD"]
    if src == "annotations":
        gid, gnames = _glomeruli_annotations(cell_type[a:b])
    else:
        gid, gnames = _glomeruli_fallback(root_ids[perm][a:b], seed)
    pops = Populations(
        order=list(POPULATION_ORDER), ranges=ranges, root_ids=root_ids[perm],
        nt=df["nt"].to_numpy(dtype=object)[perm].astype(str), flybrain_group=df["flybrain_group"].to_numpy()[perm],
        flybrain_group_names=list(FLYBRAIN_GROUP_NAMES), glomerulus_of_orn=gid, glomerulus_names=gnames,
        canonical_index=perm.astype(np.int64), source=src, cell_type=cell_type,
    )
    n_kc = pops.n("KC")
    n_mbon = pops.n("MBON_APP") + pops.n("MBON_AV") + pops.n("MBON_OTHER")
    if n_kc != EXPECTED_KC or n_mbon != EXPECTED_MBON:
        raise RuntimeError(f"population sanity check failed: KC={n_kc} (expected {EXPECTED_KC}), "
                           f"MBON={n_mbon} (expected {EXPECTED_MBON}); source={src}")
    return pops
