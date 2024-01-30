# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for extracting identifiers from MSA sequence descriptions."""

import dataclasses
import os
import pickle
import re
from typing import Optional
from absl import logging

# Sequences coming from UniProtKB database come in the
# `db|UniqueIdentifier|EntryName` format, e.g. `tr|A0A146SKV9|A0A146SKV9_FUNHE`
# or `sp|P0C2L1|A3X1_LOXLA` (for TREMBL/Swiss-Prot respectively).
_UNIPROT_PATTERN = re.compile(
    r"""
    ^
    # UniProtKB/TrEMBL or UniProtKB/Swiss-Prot
    (?:tr|sp)
    \|
    # A primary accession number of the UniProtKB entry.
    (?P<AccessionIdentifier>[A-Za-z0-9]{6,10})
    # Occasionally there is a _0 or _1 isoform suffix, which we ignore.
    (?:_\d)?
    \|
    # TREMBL repeats the accession ID here. Swiss-Prot has a mnemonic
    # protein ID code.
    (?:[A-Za-z0-9]+)
    _
    # A mnemonic species identification code.
    (?P<SpeciesIdentifier>([A-Za-z0-9]){1,5})
    # Small BFD uses a final value after an underscore, which we ignore.
    (?:_\d+)?
    $
    """,
    re.VERBOSE)


@dataclasses.dataclass(frozen=True)
class Identifiers:
  species_id: str = ''

def create_accession_seq_id_mapping(uniprot_db):
  output_path = os.path.join(os.path.dirname(uniprot_db), 'accession_seq_id_mapping.pkl')
  if not os.path.exists(output_path):
    logging.info("Creating accession species mapping from uniprot database. This can take several minutes.")
    accession_seq_id_mapping = {}


    chunk_size = 2**22

    with open(uniprot_db, 'r') as f:
        accession_seq_id_mapping = {}
        line_count = 0
        no_match_count = 0
        match_count = 0
        line_count_match = 0
        while True:
            lines = f.readlines(2**22)

            if not lines:
                #print(lines)
                #print(f'break after {line_count}')
                break

            for l in lines:
                line_count += 1
                if l.startswith('>'):
                    line_count_match += 1
                    sequence_identifier = l[1:].split()[0]
                    matches = re.search(_UNIPROT_PATTERN, sequence_identifier.strip())
                    if matches:
                        accession_seq_id_mapping[matches.group('AccessionIdentifier')] = matches.group('SpeciesIdentifier')
                        match_count += 1
                    else:
                      no_match_count += 1
            #print(f"Dict size {len(accession_seq_id_mapping)}")
            #print(f"Lines with > found: {line_count_match}, No matches: {no_match_count}, Matches: {match_count}")
    with open(output_path, 'wb') as pkl_file:
      logging.info(f"Saving to {output_path}")
      pickle.dump(accession_seq_id_mapping, pkl_file)
  else:
    with open(output_path, 'rb') as pkl_file:
       accession_seq_id_mapping = pickle.load(pkl_file)
    logging.info(f"Loaded accession species mapping with length {len(accession_seq_id_mapping)} from {output_path}")
  return accession_seq_id_mapping

def get_identifiers_mmseqs(description: str, accession_seq_id_mapping) -> Identifiers:
  """Computes extra MSA features from the description."""
  m = re.search(r'^(?:uniref90_|UniRef90_)?(\w+)\s+', description)
  if m:
    accession = m.group(1)
    try:
      sequence_identifier = accession_seq_id_mapping[accession]
    except KeyError:
      logging.warning(f"{accession} not found in accession dict")
      sequence_identifier = None
  else:
     logging.warning(f"No match in {description}")
     sequence_identifier = None
  if sequence_identifier is None:
    return Identifiers()
  else:
    return Identifiers(species_id=sequence_identifier)



def _parse_sequence_identifier(msa_sequence_identifier: str) -> Identifiers:
  """Gets species from an msa sequence identifier.

  The sequence identifier has the format specified by
  _UNIPROT_TREMBL_ENTRY_NAME_PATTERN or _UNIPROT_SWISSPROT_ENTRY_NAME_PATTERN.
  An example of a sequence identifier: `tr|A0A146SKV9|A0A146SKV9_FUNHE`

  Args:
    msa_sequence_identifier: a sequence identifier.

  Returns:
    An `Identifiers` instance with species_id. These
    can be empty in the case where no identifier was found.
  """
  matches = re.search(_UNIPROT_PATTERN, msa_sequence_identifier.strip())
  if matches:
    return Identifiers(
        species_id=matches.group('SpeciesIdentifier'))
  return Identifiers()


def _extract_sequence_identifier(description: str) -> Optional[str]:
  """Extracts sequence identifier from description. Returns None if no match."""
  split_description = description.split()
  if split_description:
    return split_description[0].partition('/')[0]
  else:
    return None


def get_identifiers(description: str) -> Identifiers:
  """Computes extra MSA features from the description."""
  sequence_identifier = _extract_sequence_identifier(description)
  if sequence_identifier is None:
    return Identifiers()
  else:
    return _parse_sequence_identifier(sequence_identifier)
