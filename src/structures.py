"""PDB parsing, chain-role assignment, and mutation application.

Two findings from the EDA are encoded here as hard rules:

* Every stated wild-type residue is present in its structure -- 2,109 of 2,109 mutated positions
  verified, zero mismatches -- so this module *asserts* agreement rather than tolerating drift.
  A mismatch means the parser broke, not that the data is dirty.
* Chain roles must be read from the structure, never from the "Protein 1" / "Protein 2" columns.
  Those columns contradict the chain groups in "#Pdb" for 2BDN_HL_A, where Protein 1 is the
  antigen yet corresponds to chains H and L.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from .paths import ensure_pdbs

THREE2ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",  # selenomethionine, deposited as HETATM; chemically methionine
}

ResKey = tuple  # (residue number: int, insertion code: str) -- antibodies need the icode

# SKEMPI mutation token: <wt><chain><resnum><icode?><mut>, e.g. YH33A or TL30aS
MUT_RE = re.compile(r"^([A-Z])([A-Za-z])(-?\d+)([a-zA-Z]?)([A-Z])$")

# Conserved immunoglobulin framework motifs, read straight off the sequence, so they work even
# where the protein-name columns are wrong or absent.
IG_MOTIFS = (
    re.compile(r"W[IVLFY][RKQ]Q"),   # FR2 tryptophan motif, heavy and light
    re.compile(r"Y[YFHC]C"),         # pre-CDR3 cysteine
    re.compile(r"WG[QKRAE]G"),       # FR4 WGxG
)

BACKBONE = ("N", "CA", "C")


@dataclass
class Mutation:
    wt: str
    chain: str
    resnum: int
    icode: str
    mut: str

    @property
    def key(self) -> ResKey:
        return (self.resnum, self.icode)

    def __str__(self) -> str:
        return f"{self.wt}{self.chain}{self.resnum}{self.icode}{self.mut}"


@dataclass
class Chain:
    id: str
    seq: str = ""
    keys: list = field(default_factory=list)    # residue keys, in sequence order
    index: dict = field(default_factory=dict)   # residue key -> offset into seq
    backbone: np.ndarray = field(default_factory=lambda: np.zeros((0, 3, 3), np.float32))
    heavy: list = field(default_factory=list)   # per residue, (n_atoms, 3) heavy-atom coords

    def __len__(self) -> int:
        return len(self.seq)


@dataclass
class Structure:
    pdb: str
    chains: dict

    def residue(self, chain: str, key: ResKey):
        ch = self.chains.get(chain)
        if ch is None:
            return None
        i = ch.index.get(key)
        return None if i is None else ch.seq[i]

    def group_seq(self, chain_group: str) -> str:
        """Concatenate the sequences of a #Pdb chain group such as 'HL'."""
        return "".join(self.chains[c].seq for c in chain_group if c in self.chains)


def parse_pdb(path: Path) -> Structure:
    """Parse one PDB file into per-chain sequences, residue indices and coordinates.

    Only the first model and the blank/'A' altloc are kept, which is what the EDA verified
    against. MSE is read from HETATM and treated as methionine; every other HETATM (waters,
    ions, ligands) is skipped.
    """
    raw = {}     # chain -> reskey -> {atom name: xyz}
    order = {}   # chain -> [reskey] in file order
    names = {}   # chain -> reskey -> one-letter code

    with open(path) as fh:
        for ln in fh:
            rec = ln[:6]
            if rec == "ENDMDL":
                break  # first model only
            if rec not in ("ATOM  ", "HETATM"):
                continue
            resname = ln[17:20].strip()
            if rec == "HETATM" and resname != "MSE":
                continue
            aa = THREE2ONE.get(resname)
            if aa is None:
                continue
            if ln[16] not in (" ", "A"):
                continue  # keep one altloc only
            element, atom = ln[76:78].strip().upper(), ln[12:16].strip()
            if element == "H" or (not element and atom.lstrip("0123456789").startswith("H")):
                continue  # heavy atoms only; hydrogens are absent from most crystal structures

            ch, key = ln[21], (int(ln[22:26]), ln[26].strip())
            xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])], np.float32)

            res = raw.setdefault(ch, {})
            if key not in res:
                res[key] = {}
                order.setdefault(ch, []).append(key)
                names.setdefault(ch, {})[key] = aa
            res[key].setdefault(atom, xyz)

    chains = {}
    for ch, keys in order.items():
        c = Chain(id=ch, keys=keys)
        c.seq = "".join(names[ch][k] for k in keys)
        c.index = {k: i for i, k in enumerate(keys)}
        bb = np.full((len(keys), 3, 3), np.nan, np.float32)
        for i, k in enumerate(keys):
            atoms = raw[ch][k]
            for j, name in enumerate(BACKBONE):
                if name in atoms:
                    bb[i, j] = atoms[name]
            c.heavy.append(
                np.stack(list(atoms.values())) if atoms else np.zeros((0, 3), np.float32)
            )
        c.backbone = bb
        chains[ch] = c
    return Structure(pdb=path.stem, chains=chains)


def load_structures(pdb_ids: Iterable[str]) -> dict:
    pdb_dir = ensure_pdbs()
    return {p: parse_pdb(pdb_dir / f"{p}.pdb") for p in sorted(set(pdb_ids))}


def parse_mutations(cleaned: str) -> list:
    """Parse a SKEMPI 'Mutation(s)_cleaned' cell into Mutation records."""
    out = []
    for tok in str(cleaned).split(","):
        tok = tok.strip()
        if not tok:
            continue
        m = MUT_RE.match(tok)
        if m is None:
            raise ValueError("unparsable mutation token %r in %r" % (tok, cleaned))
        wt, chain, num, icode, mut = m.groups()
        out.append(Mutation(wt, chain, int(num), icode.strip(), mut))
    return out


def verify(structure: Structure, muts: Iterable) -> list:
    """Return a list of problems; empty means every position resolves and the wild type agrees."""
    bad = []
    for m in muts:
        found = structure.residue(m.chain, m.key)
        if found is None:
            bad.append("%s: position absent from %s" % (m, structure.pdb))
        elif found != m.wt:
            bad.append("%s: structure has %s" % (m, found))
    return bad


def ig_score(structure: Structure, chain_group: str) -> int:
    """Count conserved Ig framework motifs across a chain group. Higher means antibody."""
    return sum(
        bool(p.search(structure.chains[c].seq))
        for c in chain_group if c in structure.chains
        for p in IG_MOTIFS
    )


def assign_roles(structure: Structure, side1: str, side2: str) -> tuple:
    """Decide which #Pdb chain group is the antibody, from structure alone.

    Returns (ab_chains, ag_chains, basis). When both sides score equally the complex is left
    unresolved with basis='tie' -- the correct answer for 1DVF_AB_CD, the anti-idiotype pair,
    which is antibody bound to antibody and has no antigen at all.
    """
    s1, s2 = ig_score(structure, side1), ig_score(structure, side2)
    if s1 > s2:
        return side1, side2, "structure"
    if s2 > s1:
        return side2, side1, "structure"
    return "", "", "tie"


def apply_mutations(structure: Structure, muts: Iterable) -> dict:
    """Return {chain: mutant sequence} for every chain touched by the mutation list.

    Asserts the wild type first: a silent off-by-one here would corrupt every downstream
    embedding difference, and the EDA proved the assertion holds for all 1,131 rows.
    """
    out = {}
    for m in muts:
        ch = structure.chains[m.chain]
        i = ch.index[m.key]
        cur = out.get(m.chain, ch.seq)
        if ch.seq[i] != m.wt:
            raise AssertionError("%s %s: structure has %s" % (structure.pdb, m, ch.seq[i]))
        out[m.chain] = cur[:i] + m.mut + cur[i + 1:]
    return out
