# Copyright 2021 DeepMind Technologies Limited
# Copyright 2022 Friedrich Miescher Institute for Biomedical Research
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
#
# Modified by Georg Kempf, Friedrich Miescher Institute for Biomedical Research

"""Functions for building the input features for the AlphaFold model."""

import os
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Union
from absl import logging
from Bio.SeqUtils import seq1
from alphafold.common import residue_constants
from alphafold.data import msa_identifiers
from alphafold.data import parsers
from alphafold.data import templates
from alphafold.data.tools import hhblits
from alphafold.data.tools import hhalign
from alphafold.data.tools import hhsearch
from alphafold.data.tools import hmmsearch
from alphafold.data.tools import jackhmmer
from alphafold.data.tools import mmseqs
from alphafold.data.templates import TemplateSearchResult
from alphafold.data.parsers import Msa
from Bio.PDB import MMCIFParser
import numpy as np
import pickle
from copy import deepcopy
from contextlib import closing
from multiprocessing import Pool
from Bio import AlignIO, SeqIO, Seq
import string
import signal
import re
import json
from shutil import copyfile

# Internal import (7716).

FeatureDict = MutableMapping[str, np.ndarray]
TemplateSearcher = Union[hhsearch.HHSearch, hmmsearch.Hmmsearch]

def make_sequence_features(
    sequence: str, description: str, num_res: int) -> FeatureDict:
  """Constructs a feature dict of sequence features."""
  features = {}
  features['aatype'] = residue_constants.sequence_to_onehot(
      sequence=sequence,
      mapping=residue_constants.restype_order_with_x,
      map_unknown_to_x=True)
  features['between_segment_residues'] = np.zeros((num_res,), dtype=np.int32)
  features['domain_name'] = np.array([description.encode('utf-8')],
                                     dtype=np.object_)
  features['residue_index'] = np.array(range(num_res), dtype=np.int32)
  features['seq_length'] = np.array([num_res] * num_res, dtype=np.int32)
  features['sequence'] = np.array([sequence.encode('utf-8')], dtype=np.object_)
  return features

def make_msa_features(msas: Sequence[parsers.Msa]) -> FeatureDict:
    """Constructs a feature dict of MSA features."""
    if not msas:
        raise ValueError('At least one MSA must be provided.')

    int_msa = []
    deletion_matrix = []
    species_ids = []
    seen_sequences = set()
    for msa_index, msa in enumerate(msas):
        if not msa:
            raise ValueError(f'MSA {msa_index} must contain at least one sequence.')
        for sequence_index, sequence in enumerate(msa.sequences):
            if sequence in seen_sequences:
                continue
            seen_sequences.add(sequence)
            int_msa.append(
                [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence])
            deletion_matrix.append(msa.deletion_matrix[sequence_index])
            identifiers = msa_identifiers.get_identifiers(
                msa.descriptions[sequence_index])
            species_ids.append(identifiers.species_id.encode('utf-8'))

    num_res = len(msas[0].sequences[0])
    num_alignments = len(int_msa)
    features = {}
    features['deletion_matrix_int'] = np.array(deletion_matrix, dtype=np.int32)
    features['msa'] = np.array(int_msa, dtype=np.int32)
    features['num_alignments'] = np.array(
        [num_alignments] * num_res, dtype=np.int32)
    features['msa_species_identifiers'] = np.array(species_ids, dtype=np.object_)
    return features


def run_msa_tool(msa_runner, input_fasta_path: str, msa_out_path: str,
                 msa_format: str, use_precomputed_msas: bool,
                 max_sto_sequences: Optional[int] = None
                 ) -> Mapping[str, Any]:
    """Runs an MSA tool, checking if output already exists first."""
    msa_out_path_a3m = msa_out_path.replace(".sto", ".a3m")
    if not use_precomputed_msas or (not os.path.exists(msa_out_path) and not os.path.exists(msa_out_path_a3m)):
        logging.info(f"No MSA found in {msa_out_path} or {msa_out_path_a3m}")
        if msa_format == 'sto' and max_sto_sequences is not None:
            result = msa_runner.query(input_fasta_path, max_sto_sequences)[0]  # pytype: disable=wrong-arg-count
        else:
            result = msa_runner.query(input_fasta_path)[0]
        with open(msa_out_path, 'w') as f:
            f.write(result[msa_format])
    else:
        if os.path.exists(msa_out_path_a3m) and msa_format == 'sto':
            msa_format = 'a3m'
            msa_out_path = msa_out_path_a3m
        logging.warning('Reading MSA from file %s', msa_out_path)
        if msa_format == 'sto' and max_sto_sequences is not None:
            precomputed_msa = parsers.truncate_stockholm_msa(
                msa_out_path, max_sto_sequences)
            result = {'sto': precomputed_msa}
        else:
            with open(msa_out_path, 'r') as f:
                result = {msa_format: f.read()}
    return result


def create_precomputed_msas_mapping(precomputed_msas_path):
    description_sequence_dict = {}
    for root_dir, dirs, files in os.walk(precomputed_msas_path):
        if "uniref90_hits.sto" in files:
            with open(os.path.join(root_dir, "uniref90_hits.sto")) as f:
                lines = [line for line in f.readlines() if not line.startswith('#')]
            desc, sequence = lines[2].split()
            description_sequence_dict[desc] = sequence.replace('-', '')

    if len(description_sequence_dict) == 0:
        logging.warning(f"No precomputed MSAs found in {precomputed_msas_path}.")
    return description_sequence_dict

def copy_files(pcmsa_path, msa_output_dir, convert=False):
    known_files = ['uniref30_colabfold_envdb',
                   'small_bfd_hits',
                   'bfd_uniclust_hits',
                   'mgnify_hits',
                   'uniref90_hits',
                   'pdb_hits']
    logging.info(f"Precomputed MSAs path: {pcmsa_path}.")
    for f in os.listdir(pcmsa_path):
        if os.path.splitext(os.path.basename(f))[0] in known_files and not f.endswith(".pkl"):
            ext = os.path.splitext(os.path.basename(f))[1]
            src_path = os.path.join(pcmsa_path, f)
            target_path = os.path.join(msa_output_dir, os.path.basename(f))
            if ext == '.a3m' or convert is False:
                if not os.path.exists(target_path):
                    logging.info(f"Copying {src_path} to {target_path}")
                    copyfile(src_path, target_path)
                else:
                    logging.info(f"Not copying precomputed MSA. {target_path} already exists.")
            elif ext == '.sto' and convert is True:
                with open(src_path, 'r') as f:
                    sto_content = f.read()
                sto = parsers.deduplicate_stockholm_msa(sto_content)
                sto = parsers.remove_empty_columns_from_stockholm_msa(sto)
                a3m = parsers.convert_stockholm_to_a3m(sto)
                target_path = target_path.replace('.sto', '.a3m')
                with open(target_path, 'w') as f:
                    f.write(a3m)

        else:
            logging.warning(f"{f} not known. Not copying.")

def slice_msa(msa_file, input_sequence):
    ext = os.path.splitext(os.path.basename(msa_file))[1]
    full_sequence_start, full_sequence_end = None, None
    remove_lowercase = str.maketrans('', '', string.ascii_lowercase)
    # if ext == '.sto':
    #     format = 'stockholm'
    if ext == '.a3m':
        format = 'fasta'
    else:
        logging.error(f"{ext} not a supported file extension. Can only handle sto or a3m.")
        raise ValueError("File format not supported.")
    # if format == 'stockholm':
    #     align = AlignIO.read(msa_file, format)
    if format == 'fasta':
        align = SeqIO.parse(msa_file, format)
    first_record = [item for i, item in enumerate(align) if i == 0][0]
    sequence = first_record.seq
    #Record residue indices (excluding gaps and insertions)
    res_indices = [i for i, res in enumerate(sequence) if res != '-' and not res.islower()]
    #print(res_indices)
    #Remove gaps (-) to get continues sequence
    seq_reduced = str(sequence).replace('-', '')
    #print(seq_reduced)
    # Remove insertions (lowercase in case of a3m) to get continues sequence
    seq_reduced = seq_reduced.translate(remove_lowercase)
    #print(seq_reduced)
    match_indices = [(m.start(0), m.end(0)) for m in re.finditer(input_sequence, str(seq_reduced))]
    #Get start and end indices for the subsequence slice
    if len(match_indices) > 0:
        first_match_indices = match_indices[0]
        full_sequence_start = res_indices[first_match_indices[0]]
        full_sequence_end = res_indices[first_match_indices[1]] - 1
    else:
        logging.debug(f"{input_sequence} is not a subsequence of {seq_reduced}")
    logging.debug(f"Start index {full_sequence_start}, End index {full_sequence_end}")
    #print(align)
    if not full_sequence_start is None and not full_sequence_end is None:
        # if format == 'stockholm':
        #     #Only keep columns that correspond to subsequence
        #     new_align = align[:, full_sequence_start:full_sequence_end + 1]
        #     SeqIO.write(new_align, msa_file, format)
        if format == 'fasta':
            print(f"FORMAT FASTA")
            #a3m includes insertions that result in unequal sequence length for records. These need to be removed
            #to slice the subsequence and then re-added to the sliced sequences.
            records = [record for record in SeqIO.parse(msa_file, format)]
            with open(msa_file, 'w') as f:
                for record in records:
                    seq = str(record.seq)
                    #Record insertion positions to be re-added after column removal
                    insertions = [(i, res) for i, res in enumerate(seq) if res.islower()]
                    #Record positions of non-insertion letters
                    non_insertions = [(i, res) for i, res in enumerate(seq) if not res.islower()]
                    non_insertions_i, non_insertions_res = zip(*non_insertions)
                    #Slice sequence indices (excluding insertions)
                    non_insertions_i = non_insertions_i[full_sequence_start:full_sequence_end + 1]
                    #Slice sequence indices (excluding insertions)
                    non_insertions_res = non_insertions_res[full_sequence_start:full_sequence_end + 1]
                    #print(non_insertions_i)
                    #print(non_insertions_res)
                    #Get insertions that are within the slice range
                    insertions = [(item[0], item[1]) for item in insertions
                                  if item[0] >= non_insertions_i[0] and item[0] <= non_insertions_i[-1]]

                    #if len(insertions) > 0:
                        #print("Insertions after column removal")
                        #print(insertions)
                    insertions_added = []
                    #Re-add insertions that are within the slice
                    for i, res in enumerate(non_insertions_res):
                        index = non_insertions_i[i]
                        try:
                            next_index = non_insertions_i[i+1]
                        except IndexError:
                            next_index = index
                        insertions_added.append(res)
                        for insertion_index, insertion_res in insertions:
                            if insertion_index > index and insertion_index < next_index:
                                #print(f"Insertion index {insertion_index} is larger than {index} and smaller than {next_index}")
                                insertions_added.append(insertion_res)
                    record.seq = Seq.Seq(''.join(insertions_added))
                    SeqIO.write(record, f, format)

def is_subsequence(input_sequence, precomputed_msas_path):
    desc_seq_dict = create_precomputed_msas_mapping(precomputed_msas_path)
    for seq in desc_seq_dict.values():
        if re.search(str(input_sequence), str(seq)):
            logging.debug(f"{input_sequence} is a subsequence of {seq}")
            if len(input_sequence) < len(seq):
                return True
        else:
            logging.debug(f"{input_sequence} not a subsequence of {seq}")
    return False

def get_precomputed_msas_path(precomputed_msas_path):
    #Search for msas dir in case the job base directory is given
    for root_dir, _, _ in os.walk(precomputed_msas_path, topdown=True):
        if os.path.basename(os.path.normpath(root_dir)) == 'msas':
            precomputed_msas_path = root_dir
            break
    return precomputed_msas_path


def get_pcmsa_map(precomputed_msas_path, new_map):
    precomputed_msas_path = get_precomputed_msas_path(precomputed_msas_path)
    precomputed_chain_id_map = os.path.join(precomputed_msas_path, 'chain_id_map.json')

    if os.path.exists(precomputed_chain_id_map):
        with open(precomputed_chain_id_map, 'r') as f:
            prev_map = json.load(f)
    else:
        prev_map = create_precomputed_msas_mapping(precomputed_msas_path)

    pcmsa_map = {}
    #key = new chain_id or description
    #value = new mapping or sequence
    for key, value in new_map.items():
        #key = previous chain or description
        #value = previous mapping or sequence
        for prev_key, prev_value in prev_map.items():
            if hasattr(prev_value, 'sequence'):
                prev_sequence = prev_value.sequence
            elif 'sequence' in prev_value:
                prev_sequence = prev_value['sequence']
            else:
                prev_sequence = prev_value
            prev_folder_name = prev_key
            if hasattr(value, 'sequence'):
                sequence = value.sequence
            else:
                sequence = value
            if re.search(sequence, prev_sequence):
                #Check if previous job was monomer job
                if 'uniref90_hits.sto' in os.listdir(precomputed_msas_path):
                    prev_msa_dir = precomputed_msas_path
                elif prev_folder_name in os.listdir(precomputed_msas_path):
                    prev_msa_dir = os.path.join(precomputed_msas_path, prev_folder_name)
                else:
                    raise ValueError("No MSAs found in precomputed_msas_path.")
                pcmsa_map[key] = prev_msa_dir

    return pcmsa_map


class DataPipeline:
  """Runs the alignment tools and assembles the input features."""

  def __init__(self,
               jackhmmer_binary_path: str,
               hhblits_binary_path: str,
               mmseqs_binary_path: str,
               uniref90_database_path: str,
               mgnify_database_path: str,
               bfd_database_path: Optional[str],
               uniclust30_database_path: Optional[str],
               small_bfd_database_path: Optional[str],
               uniref30_database_path: str,
               colabfold_envdb_database_path: str,
               template_searcher: TemplateSearcher,
               template_featurizer: templates.TemplateHitFeaturizer,
               use_small_bfd: bool,
               use_mmseqs: bool,
               mgnify_max_hits: int = 501,
               uniref_max_hits: int = 10000,
               use_precomputed_msas: bool = False,
               custom_tempdir: str = None,
               precomputed_msas_path: str = None):
    """Initializes the data pipeline."""
    self._use_small_bfd = use_small_bfd
    self._use_mmseqs = use_mmseqs
    self.jackhmmer_uniref90_runner = jackhmmer.Jackhmmer(
        binary_path=jackhmmer_binary_path,
        database_path=uniref90_database_path,
        custom_tempdir=custom_tempdir)
    if use_small_bfd:
      self.jackhmmer_small_bfd_runner = jackhmmer.Jackhmmer(
          binary_path=jackhmmer_binary_path,
          database_path=small_bfd_database_path,
          custom_tempdir=custom_tempdir)
    else:
      self.hhblits_bfd_uniclust_runner = hhblits.HHBlits(
          binary_path=hhblits_binary_path,
          databases=[bfd_database_path, uniclust30_database_path],
          custom_tempdir=custom_tempdir)
    self.jackhmmer_mgnify_runner = jackhmmer.Jackhmmer(
        binary_path=jackhmmer_binary_path,
        database_path=mgnify_database_path,
        custom_tempdir=custom_tempdir)
    self.mmseqs_runner = mmseqs.MMSeqs(
        binary_path=mmseqs_binary_path,
        database_path=[uniref30_database_path, colabfold_envdb_database_path],
        custom_tempdir=custom_tempdir)
    self.template_searcher = template_searcher
    self.template_featurizer = template_featurizer
    self.mgnify_max_hits = mgnify_max_hits
    self.uniref_max_hits = uniref_max_hits
    self.use_precomputed_msas = use_precomputed_msas
    self.custom_tempdir = custom_tempdir
    self.precomputed_msas_path = precomputed_msas_path

  def get_template_sequence(self, custom_template: str):
    cifparser = MMCIFParser()
    model = cifparser.get_structure('model', custom_template)[0]
    name = os.path.splitext(os.path.basename(custom_template))[0]
    chains = [str(chain.get_id()) for chain in model.get_chains()]
    if len(chains) > 1:
        raise SystemExit("The custom template contains more than one chain!")
    else:
        chain_id = chains[0]
    sequence = seq1(''.join(residue.resname for residue in model[chain_id]))
    custom_template_sequence = {'name': name,
                                'sequence': sequence}
    return custom_template_sequence

  def init_worker(self):
      signal.signal(signal.SIGINT, signal.SIG_IGN)


  def process(self,
          input_fasta_path,
          msa_output_dir,
          no_msa,
          no_template,
          custom_template,
          precomputed_msas,
          num_cpu):
    #If number of CPU > 2 hhblits and jackhmmer jobs can run in parallel. Available CPUs (devided by two) will be forwarded to the
    #hhblits and jackhmmer "tools". In case of mmseqs job not more than 8 CPUs will be used for the jackhmmer job and the rest
    #will be used for mmseqs.
    mmseqs_cpu, tool_cpu = 1, 1
    if num_cpu > 2:
      if self._use_mmseqs:
          if num_cpu > 16:
              tool_cpu = 8
              mmseqs_cpu = num_cpu - 8
          else:
              tool_cpu = int(num_cpu / 2)
              mmseqs_cpu = int(num_cpu / 2)
      else:
          tool_cpu = int(num_cpu / 2)

    with open(input_fasta_path) as f:
        input_fasta_str = f.read()
    input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
    if len(input_seqs) != 1:
        raise ValueError(
            f'More than one input sequence found in {input_fasta_path}.')
    input_sequence = input_seqs[0]
    input_description = input_descs[0]
    num_res = len(input_sequence)

    if precomputed_msas:
        if isinstance(precomputed_msas, list):
            if len(precomputed_msas) > 1:
                raise ValueError("Too many items at this stage")
            else:
                precomputed_msas = precomputed_msas[0]
        if not precomputed_msas in [None, "None", "none"]:
            if is_subsequence(input_sequence, precomputed_msas):
                copy_files(precomputed_msas, msa_output_dir, convert=True)
                logging.info("Input sequence is a subsequence of provided MSAs. MSAs will be cropped.")
                for msa_file in ['uniref30_colabfold_envdb.a3m',
                            'small_bfd_hits.a3m',
                            'uniref90_hits.a3m',
                            'bfd_uniclust_hits.a3m',
                            'mgnify_hits.a3m']:
                    msa_file = os.path.join(msa_output_dir, msa_file)
                    if os.path.exists(msa_file):
                        slice_msa(msa_file, input_sequence)
            else:
                copy_files(precomputed_msas, msa_output_dir, convert=False)
                logging.info("Not a subsequence.")
    self.mmseqs_runner.n_cpu = mmseqs_cpu
    self.jackhmmer_uniref90_runner.n_cpu = tool_cpu
    self.jackhmmer_mgnify_runner.n_cpu = tool_cpu
    if self._use_small_bfd:
        self.jackhmmer_small_bfd_runner.n_cpu = tool_cpu
    else:
        self.hhblits_bfd_uniclust_runner.n_cpu = tool_cpu
    logging.info(f"No MSA: {no_msa}; No Templates: {no_template}; Custom Template {custom_template}; Precomputed MSAs {precomputed_msas}")
    if isinstance(no_msa, list):
        if len(no_msa) > 1:
            raise ValueError("Too many items at this stage")
        else:
            no_msa = no_msa[0]

    if isinstance(no_template, list):
        if len(no_template) > 1:
            raise ValueError("Too many items at this stage")
        else:
            no_template = no_template[0]


    if isinstance(custom_template, list):
         if len(custom_template) > 1:
             raise ValueError("Too many items at this stage")
         else:
             custom_template = custom_template[0]


    """Runs alignment tools on the input sequence and creates features."""
    msa_jobs = []
    if not no_msa:
        if self._use_mmseqs:
            uniref30_colabfold_envdb_out_path = os.path.join(msa_output_dir, 'uniref30_colabfold_envdb.a3m')
            msa_jobs.append(
                (self.mmseqs_runner,
                 input_fasta_path,
                 uniref30_colabfold_envdb_out_path,
                 'a3m',
                 self.use_precomputed_msas))
        else:
            if self._use_small_bfd:
                bfd_out_path = os.path.join(msa_output_dir, 'small_bfd_hits.sto')
                msa_jobs.append((
                    self.jackhmmer_small_bfd_runner,
                    input_fasta_path,
                    bfd_out_path,
                    'sto',
                    self.use_precomputed_msas))

            else:
                bfd_out_path = os.path.join(msa_output_dir, 'bfd_uniclust_hits.a3m')
                msa_jobs.append((
                    self.hhblits_bfd_uniclust_runner,
                    input_fasta_path,
                    bfd_out_path,
                    'a3m',
                    self.use_precomputed_msas))

            mgnify_out_path = os.path.join(msa_output_dir, 'mgnify_hits.sto')
            msa_jobs.append((self.jackhmmer_mgnify_runner,
            input_fasta_path,
            mgnify_out_path,
            'sto',
            self.use_precomputed_msas,
            self.mgnify_max_hits))

    #uniref90 msa also needed for PDB hit search and custom template
    if not no_msa or not no_template:
        uniref90_out_path = os.path.join(msa_output_dir, 'uniref90_hits.sto')
        msa_jobs.append((
            self.jackhmmer_uniref90_runner,
            input_fasta_path,
            uniref90_out_path,
            'sto',
            self.use_precomputed_msas,
            self.uniref_max_hits))

    logging.info("Job list")
    logging.info(msa_jobs)
    if not len(msa_jobs) == 0:
        with closing(Pool(2, self.init_worker)) as pool:
            try:
                results = pool.starmap_async(run_msa_tool, msa_jobs)
                msa_jobs_results = results.get()
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()

    if not no_msa:
        if self._use_mmseqs:
            mmseqs_uniref30_colabfold_envdb_result = msa_jobs_results[0]
            uniref30_colabfold_envdb_msa = parsers.parse_a3m(mmseqs_uniref30_colabfold_envdb_result['a3m'])
        else:
            if self._use_small_bfd:
                jackhmmer_small_bfd_result = msa_jobs_results[0]
                if 'sto' in jackhmmer_small_bfd_result:
                    bfd_msa = parsers.parse_stockholm(jackhmmer_small_bfd_result['sto'])
                elif 'a3m' in jackhmmer_small_bfd_result:
                    bfd_msa = parsers.parse_a3m(jackhmmer_small_bfd_result['a3m'])
                else:
                    raise ValueError("Format not known.")
            else:
                hhblits_bfd_uniclust_result = msa_jobs_results[0]
                bfd_msa = parsers.parse_a3m(hhblits_bfd_uniclust_result['a3m'])
    
            jackhmmer_mgnify_result = msa_jobs_results[1]
            if 'sto' in jackhmmer_mgnify_result:
                mgnify_msa = parsers.parse_stockholm(jackhmmer_mgnify_result['sto'])
            elif 'a3m' in jackhmmer_mgnify_result:
                mgnify_msa = parsers.parse_a3m(jackhmmer_mgnify_result['a3m'])
            mgnify_msa = mgnify_msa.truncate(max_seqs=self.mgnify_max_hits)


    if not no_msa or not no_template:
        if len(msa_jobs) == 1:
            jackhmmer_uniref90_result = msa_jobs_results[0]
        else:
            jackhmmer_uniref90_result = msa_jobs_results[-1]
        if not 'a3m' in jackhmmer_uniref90_result:
            uniref90_msa = parsers.parse_stockholm(jackhmmer_uniref90_result['sto'])
            uniref90_msa = uniref90_msa.truncate(max_seqs=self.uniref_max_hits)
            msa_for_templates = jackhmmer_uniref90_result['sto']
            msa_for_templates = parsers.deduplicate_stockholm_msa(msa_for_templates)
            msa_for_templates = parsers.remove_empty_columns_from_stockholm_msa(
                msa_for_templates)
        else:
            uniref90_msa = parsers.parse_a3m(jackhmmer_uniref90_result['a3m'])
            msa_for_templates = jackhmmer_uniref90_result


    #Refactored to implement use of a single custom template
    if not custom_template is None:
        logging.info("Using custom template")
        if os.path.exists(custom_template):
            logging.info(custom_template)
            template_sequence = self.get_template_sequence(custom_template)
            template_searcher = hhalign.HHAlign(self.template_searcher.hhalign_binary_path)
            if not 'a3m' in msa_for_templates:
                uniref90_msa_as_a3m = parsers.convert_stockholm_to_a3m(msa_for_templates)
            else:
                uniref90_msa_as_a3m = msa_for_templates['a3m']
            pdb_templates_result = template_searcher.query(template_sequence, uniref90_msa_as_a3m)

            self.template_featurizer_initial = deepcopy(self.template_featurizer)
            self.template_featurizer = templates.HhsearchHitFeaturizer(
                mmcif_dir=os.path.dirname(custom_template),
                max_template_date=self.template_featurizer_initial._max_template_date.strftime("%Y-%m-%d"),
                max_hits=self.template_featurizer_initial._max_hits,
                kalign_binary_path=self.template_featurizer_initial._kalign_binary_path,
                release_dates_path=None,
                obsolete_pdbs_path=None,
                strict_error_check=False,
                custom_tempdir=self.custom_tempdir)
            pdb_template_hits = template_searcher.get_template_hits(
                output_string=pdb_templates_result, input_sequence=input_sequence)
            logging.info(pdb_template_hits)
        else:
            logging.error(f"Custom template not found in {custom_template}")
    elif not no_template:
        logging.info("Using templates from the PDB")
        pdb_hits_out_path = os.path.join(
            msa_output_dir, f'pdb_hits.{self.template_searcher.output_format}')
        if not os.path.exists(pdb_hits_out_path):
            if 'a3m' in msa_for_templates:
                pdb_templates_result = self.template_searcher.query(msa_for_templates['a3m'], format='a3m')
            elif self.template_searcher.input_format == 'sto' and not 'a3m' in msa_for_templates:
                pdb_templates_result = self.template_searcher.query(msa_for_templates)
            elif self.template_searcher.input_format == 'a3m':
                uniref90_msa_as_a3m = parsers.convert_stockholm_to_a3m(msa_for_templates)
                pdb_templates_result = self.template_searcher.query(uniref90_msa_as_a3m)
            else:
                raise ValueError('Unrecognized template input format: '
                                 f'{self.template_searcher.input_format}')
            with open(pdb_hits_out_path, 'w') as f:
                f.write(pdb_templates_result)
        else:
            with open(pdb_hits_out_path, 'r') as f:
                pdb_templates_result = f.read()
        pdb_template_hits = self.template_searcher.get_template_hits(
            output_string=pdb_templates_result, input_sequence=input_sequence)



    template_result_out = os.path.join(
        msa_output_dir, 'template_results.pkl')
    if no_template:
        logging.info("Generating dummy template.")
        templates_result = TemplateSearchResult(features={
            'template_aatype': np.zeros(
                (1, num_res, len(residue_constants.restypes_with_x_and_gap)),
                np.float32),
            'template_all_atom_masks': np.zeros(
                (1, num_res, residue_constants.atom_type_num), np.float32),
            'template_all_atom_positions': np.zeros(
                (1, num_res, residue_constants.atom_type_num, 3), np.float32),
            'template_domain_names': np.array([''.encode()], dtype=np.object),
            'template_sequence': np.array([''.encode()], dtype=np.object),
            'template_sum_probs': np.array([0], dtype=np.float32)
        }, errors=[], warnings=[])
    elif all([os.path.exists(template_result_out),
              #not len(input_sequence) < len(uniref90_msa.sequences[0]),
              not custom_template,
              not no_template,
              self.use_precomputed_msas]):
        logging.info("Opening template from pickle")
        with open(template_result_out, 'rb') as f:
            templates_result = pickle.load(f)
    else:
        logging.info("Get template from pdb hits or custom file.")
        templates_result = self.template_featurizer.get_templates(
            query_sequence=input_sequence,
            hits=pdb_template_hits)
        with open(template_result_out, 'wb') as f:
            pickle.dump(templates_result, f)

    sequence_features = make_sequence_features(
        sequence=input_sequence,
        description=input_description,
        num_res=num_res)

    logging.info(no_msa)
    if no_msa:
        empty_msa = Msa(sequences=[input_sequence],
            deletion_matrix=[[0 for _ in range(len(input_sequence))]],
            descriptions=[input_description])
        msa_features = make_msa_features((empty_msa, empty_msa, empty_msa))
        logging.info("Not using MSA")
    else:
        logging.info("Using MSA")
        if self._use_mmseqs:
            msa_features = make_msa_features((uniref90_msa, uniref30_colabfold_envdb_msa))
        else:
            msa_features = make_msa_features((uniref90_msa, bfd_msa, mgnify_msa))

    if not no_msa:
        logging.info('Uniref90 MSA size: %d sequences.', len(uniref90_msa))
        if self._use_mmseqs:
            logging.info('Uniref30 and colabfold_envdb MSA size: %d sequences.', len(uniref30_colabfold_envdb_msa))
        else:
            logging.info('BFD MSA size: %d sequences.', len(bfd_msa))
            logging.info('MGnify MSA size: %d sequences.', len(mgnify_msa))
            logging.info('Final (deduplicated) MSA size: %d sequences.',
                         msa_features['num_alignments'][0])
    else:
        logging.info('Final (deduplicated) MSA size: %d empty sequence (as requested).',
                     msa_features['num_alignments'][0])
    if not no_template:
        logging.info('Total number of templates (NB: this can include bad '
                     'templates and is later filtered to top 4): %d.',
                     templates_result.features['template_domain_names'].shape[0])
    else:
        logging.info('Total number of templates (NB: this can include bad '
                     'templates and is later filtered to top 4): %d empty dummy template (as requested).',
                     templates_result.features['template_domain_names'].shape[0])


    return {**sequence_features, **msa_features, **templates_result.features}
