#!/usr/bin/env python3
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

#Modified by Georg Kempf, Friedrich Miescher Institute for Biomedical Research

"""Full AlphaFold protein structure prediction script."""
import json
import os
import pathlib
import pickle
import random
import shutil
import sys
import time
import traceback
from typing import Dict, Union, Optional

from absl import app
from absl import flags
from absl import logging
from guifold.afeval import EvaluationPipeline
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.data import pipeline
from alphafold.data import pipeline_multimer
from alphafold.data import templates
from alphafold.data.tools import hhsearch
from alphafold.data.tools import hmmsearch
from alphafold.model import config
from alphafold.model import model
from alphafold.relax import relax
from alphafold.data import parsers
import numpy as np
import inspect

from alphafold.model import data
# Internal import (7716).

logging.set_verbosity(logging.INFO)


flags.DEFINE_string('fasta_path', None, 'Path to a single fasta file.')
flags.DEFINE_string('data_dir', None, 'Path to directory of supporting data.')
flags.DEFINE_string('output_dir', None, 'Path to a directory that will '
                    'store the results.')
flags.DEFINE_string('jackhmmer_binary_path', None,
                    'Path to the JackHMMER executable.')
flags.DEFINE_string('hhblits_binary_path', None,
                    'Path to the HHblits executable.')
flags.DEFINE_string('hhsearch_binary_path', None,
                    'Path to the HHsearch executable.')
flags.DEFINE_string('hmmsearch_binary_path', None,
                    'Path to the hmmsearch executable.')
flags.DEFINE_string('hmmbuild_binary_path', None,
                    'Path to the hmmbuild executable.')
flags.DEFINE_string('hhalign_binary_path', None,
                    'Path to the hhalign executable.')
flags.DEFINE_string('kalign_binary_path', None,
                    'Path to the Kalign executable.')
flags.DEFINE_string('uniref90_database_path', None, 'Path to the Uniref90 '
                    'database for use by JackHMMER.')
flags.DEFINE_string('mgnify_database_path', None, 'Path to the MGnify '
                    'database for use by JackHMMER.')
flags.DEFINE_string('bfd_database_path', None, 'Path to the BFD '
                    'database for use by HHblits.')
flags.DEFINE_string('small_bfd_database_path', None, 'Path to the small '
                    'version of BFD used with the "reduced_dbs" preset.')
flags.DEFINE_string('uniclust30_database_path', None, 'Path to the Uniclust30 '
                    'database for use by HHblits.')
flags.DEFINE_string('uniprot_database_path', None, 'Path to the Uniprot '
                    'database for use by JackHMMer.')
flags.DEFINE_string('pdb70_database_path', None, 'Path to the PDB70 '
                    'database for use by HHsearch.')
flags.DEFINE_string('pdb_seqres_database_path', None, 'Path to the PDB '
                    'seqres database for use by hmmsearch.')
flags.DEFINE_string('template_mmcif_dir', None, 'Path to a directory with '
                    'template mmCIF structures, each named <pdb_id>.cif')
flags.DEFINE_string('max_template_date', None, 'Maximum template release date '
                    'to consider. Important if folding historical test sets.')
flags.DEFINE_string('obsolete_pdbs_path', None, 'Path to file containing a '
                    'mapping from obsolete PDB IDs to the PDB IDs of their '
                    'replacements.')
flags.DEFINE_enum('db_preset', 'full_dbs',
                  ['full_dbs', 'reduced_dbs'],
                  'Choose preset MSA database configuration - '
                  'smaller genetic database config (reduced_dbs) or '
                  'full genetic database config  (full_dbs)')
flags.DEFINE_enum('model_preset', 'monomer',
                  ['monomer', 'monomer_casp14', 'monomer_ptm', 'multimer'],
                  'Choose preset model configuration - the monomer model, '
                  'the monomer model with extra ensembling, monomer model with '
                  'pTM head, or multimer model')
flags.DEFINE_boolean('benchmark', False, 'Run multiple JAX model evaluations '
                     'to obtain a timing that excludes the compilation time, '
                     'which should be more indicative of the time required for '
                     'inferencing many proteins.')
flags.DEFINE_integer('random_seed', None, 'The random seed for the data '
                     'pipeline. By default, this is randomly generated. Note '
                     'that even if this is set, Alphafold may still not be '
                     'deterministic, because processes like GPU inference are '
                     'nondeterministic.')
flags.DEFINE_integer('num_multimer_predictions_per_model', 1, 'How many '
                     'predictions (each with a different random seed) will be '
                     'generated per model. E.g. if this is 2 and there are 5 '
                     'models then there will be 10 predictions per input. '
                     'Note: this FLAG only applies if model_preset=multimer')
flags.DEFINE_boolean('use_precomputed_msas', False, 'Whether to read MSAs that '
                     'have been written to disk instead of running the MSA '
                     'tools. The MSA files are looked up in the output '
                     'directory, so it must stay the same between multiple '
                     'runs that are to reuse the MSAs. WARNING: This will not '
                     'check if the sequence, database or configuration have '
                     'changed.')
flags.DEFINE_boolean('run_relax', True, 'Whether to run the final relaxation '
                     'step on the predicted models. Turning relax off might '
                     'result in predictions with distracting stereochemical '
                     'violations but might help in case you are having issues '
                     'with the relaxation stage.')
flags.DEFINE_boolean('use_gpu_relax', True, 'Whether to relax on GPU. '
                     'Relax on GPU can be much faster than CPU, so it is '
                     'recommended to enable if possible. GPUs must be available'
                     ' if this setting is enabled.')
flags.DEFINE_list('no_msa_list', False, 'Optional. If the use of MSAs should be disabled for a sequence'
                                   'a boolean needs to be given for each sequence in the same order '
                                   'as sequences are given in the fasta file.')
flags.DEFINE_list('no_template_list', False, 'Optional. If the use of templates should be disabled for a sequence'
                                                'a boolean needs to be given for each sequence in the same order '
                                                'as sequences are given in the fasta file.')
flags.DEFINE_list('custom_template_list', None, 'Optional. If a custom template should be used for one or'
                                                    ' more sequences, a list of file paths or None needs to be given  '
                                                'in the same order as sequences are given in the fasta file.')
flags.DEFINE_string('custom_tempdir', None, 'Define a custom tempdir other than /tmp')
flags.DEFINE_integer('num_recycle', 3, 'Define maximum number of model recycles.')
flags.DEFINE_bool('only_features', False, 'Stop after Feature pipeline. Useful for splitting up the job into CPU and GPU resources.')
flags.DEFINE_bool('continue_from_features', False, 'Continue from features.pkl file.'
                                                   ' Useful for splitting up the job into CPU and GPU resources.')
flags.DEFINE_string('scratch_dir', None, 'Directory to temporarily store database index (for use with MMSeqs2),'
                                          ' e.g. RAM or local SSD.')

FLAGS = flags.FLAGS

MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3


def _check_flag(flag_name: str,
                other_flag_name: str,
                should_be_set: bool):
  if should_be_set != bool(FLAGS[flag_name].value):
    verb = 'be' if should_be_set else 'not be'
    raise ValueError(f'{flag_name} must {verb} set when running with '
                     f'"--{other_flag_name}={FLAGS[other_flag_name].value}".')


def predict_structure(
    fasta_path: str,
    fasta_name: str,
    output_dir_base: str,
    data_pipeline: Union[pipeline.DataPipeline, pipeline_multimer.DataPipeline],
    model_runners: Dict[str, model.RunModel],
    amber_relaxer: relax.AmberRelaxation,
    benchmark: bool,
    random_seed: int,
    no_msa_list: Optional[bool] = None,
    no_template_list: Optional[bool] = None,
    custom_template_list: Optional[str] = None):
  """Predicts structure using AlphaFold for the given sequence."""
  logging.info('Predicting %s', fasta_name)
  timings = {}
  output_dir = os.path.join(output_dir_base, fasta_name)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  msa_output_dir = os.path.join(output_dir, 'msas')
  if not os.path.exists(msa_output_dir):
    os.makedirs(msa_output_dir)

  features_output_path = os.path.join(output_dir, 'features.pkl')
  # Get features.
  if not FLAGS.continue_from_features:
      t_0 = time.time()
      feature_dict = data_pipeline.process(
            input_fasta_path=fasta_path,
            msa_output_dir=msa_output_dir,
            no_msa=no_msa_list,
            no_template=no_template_list,
            custom_template=custom_template_list)
      timings['features'] = time.time() - t_0

      # Write out features as a pickled dictionary.

      with open(features_output_path, 'wb') as f:
        pickle.dump(feature_dict, f, protocol=4)

  #Stop here if only_msa flag is set
  if not FLAGS.only_features:
      if FLAGS.continue_from_features:
          if not os.path.exists(features_output_path):
              raise("Continue_from_features requested but no feature pickle file found in this directory.")
          with open(features_output_path, 'rb') as f:
            feature_dict = pickle.load(f)

      unrelaxed_pdbs = {}
      relaxed_pdbs = {}
      ranking_confidences = {}
    
      # Run the models.
      num_models = len(model_runners)
      for model_index, (model_name, model_runner) in enumerate(
          model_runners.items()):
        logging.info('Running model %s on %s', model_name, fasta_name)
        t_0 = time.time()
        model_random_seed = model_index + random_seed * num_models
        processed_feature_dict = model_runner.process_features(
            feature_dict, random_seed=model_random_seed)
        timings[f'process_features_{model_name}'] = time.time() - t_0
    
        t_0 = time.time()
        prediction_result = model_runner.predict(processed_feature_dict,
                                                 random_seed=model_random_seed)
        t_diff = time.time() - t_0
        timings[f'predict_and_compile_{model_name}'] = t_diff
        logging.info(
            'Total JAX model %s on %s predict time (includes compilation time, see --benchmark): %.1fs',
            model_name, fasta_name, t_diff)
    
        if benchmark:
          t_0 = time.time()
          model_runner.predict(processed_feature_dict,
                               random_seed=model_random_seed)
          t_diff = time.time() - t_0
          timings[f'predict_benchmark_{model_name}'] = t_diff
          logging.info(
              'Total JAX model %s on %s predict time (excludes compilation time): %.1fs',
              model_name, fasta_name, t_diff)
    
        plddt = prediction_result['plddt']
        ranking_confidences[model_name] = prediction_result['ranking_confidence']
    
        # Save the model outputs.
        result_output_path = os.path.join(output_dir, f'result_{model_name}.pkl')
        with open(result_output_path, 'wb') as f:
          pickle.dump(prediction_result, f, protocol=4)
    
        # Add the predicted LDDT in the b-factor column.
        # Note that higher predicted LDDT value means higher model confidence.
        plddt_b_factors = np.repeat(
            plddt[:, None], residue_constants.atom_type_num, axis=-1)
        unrelaxed_protein = protein.from_prediction(
            features=processed_feature_dict,
            result=prediction_result,
            b_factors=plddt_b_factors,
            remove_leading_feature_dimension=not model_runner.multimer_mode)
    
        unrelaxed_pdbs[model_name] = protein.to_pdb(unrelaxed_protein)
        unrelaxed_pdb_path = os.path.join(output_dir, f'unrelaxed_{model_name}.pdb')
        with open(unrelaxed_pdb_path, 'w') as f:
          f.write(unrelaxed_pdbs[model_name])
    
        if amber_relaxer:
          # Relax the prediction.
          t_0 = time.time()
          relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)
          timings[f'relax_{model_name}'] = time.time() - t_0
    
          relaxed_pdbs[model_name] = relaxed_pdb_str
    
          # Save the relaxed PDB.
          relaxed_output_path = os.path.join(
              output_dir, f'relaxed_{model_name}.pdb')
          with open(relaxed_output_path, 'w') as f:
            f.write(relaxed_pdb_str)
    
      # Rank by model confidence and write out relaxed PDBs in rank order.
      ranked_order = []
      for idx, (model_name, _) in enumerate(
          sorted(ranking_confidences.items(), key=lambda x: x[1], reverse=True)):
        ranked_order.append(model_name)
        ranked_output_path = os.path.join(output_dir, f'ranked_by_plddt_{idx}.pdb')
        with open(ranked_output_path, 'w') as f:
          if amber_relaxer:
            f.write(relaxed_pdbs[model_name])
          else:
            f.write(unrelaxed_pdbs[model_name])
    
      ranking_output_path = os.path.join(output_dir, 'ranking_debug.json')
      with open(ranking_output_path, 'w') as f:
        label = 'iptm+ptm' if 'iptm' in prediction_result else 'plddts'
        f.write(json.dumps(
            {label: ranking_confidences, 'order': ranked_order}, indent=4))
    
      logging.info('Final timings for %s: %s', fasta_name, timings)
    
      timings_output_path = os.path.join(output_dir, 'timings.json')
      with open(timings_output_path, 'w') as f:
        f.write(json.dumps(timings, indent=4))

      evaluation = EvaluationPipeline(FLAGS.fasta_path)
      evaluation.run_pipeline()
      logging.info("Alphafold pipeline completed. Exit code 0")
  else:
      logging.info("Alphafold pipeline completed with feature generation. Exit code 0")


def parse_fasta(fasta_path):
    with open(fasta_path) as f:
        input_fasta_str = f.read()
    input_seqs, _ = parsers.parse_fasta(input_fasta_str)
    return input_seqs

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  #Do not check for MSA tools when MSA already exists.
  run_multimer_system = 'multimer' in FLAGS.model_preset
  use_small_bfd = FLAGS.db_preset == 'reduced_dbs'
  if not FLAGS.use_precomputed_msas:
      for tool_name in (
          'jackhmmer', 'hhblits', 'hhsearch', 'hmmsearch', 'hmmbuild', 'kalign'):
        if not FLAGS[f'{tool_name}_binary_path'].value:
          raise ValueError(f'Could not find path to the "{tool_name}" binary. Make '
                           'sure it is installed on your system.')
      _check_flag('small_bfd_database_path', 'db_preset',
                  should_be_set=use_small_bfd)
      _check_flag('bfd_database_path', 'db_preset',
                  should_be_set=not use_small_bfd)
      _check_flag('uniclust30_database_path', 'db_preset',
                  should_be_set=not use_small_bfd)
      _check_flag('pdb70_database_path', 'model_preset',
                  should_be_set=not run_multimer_system)
      _check_flag('pdb_seqres_database_path', 'model_preset',
                  should_be_set=run_multimer_system)
      _check_flag('uniprot_database_path', 'model_preset',
                  should_be_set=run_multimer_system)

  if FLAGS.model_preset == 'monomer_casp14':
    num_ensemble = 8
  else:
    num_ensemble = 1

  #Only one fasta file allowed
  fasta_name = pathlib.Path(FLAGS.fasta_path).stem

  if run_multimer_system:
    template_searcher = hmmsearch.Hmmsearch(
        binary_path=FLAGS.hmmsearch_binary_path,
        hmmbuild_binary_path=FLAGS.hmmbuild_binary_path,
        hhalign_binary_path=FLAGS.hhalign_binary_path,
        database_path=FLAGS.pdb_seqres_database_path)
    template_featurizer = templates.HmmsearchHitFeaturizer(
        mmcif_dir=FLAGS.template_mmcif_dir,
        max_template_date=FLAGS.max_template_date,
        max_hits=MAX_TEMPLATE_HITS,
        kalign_binary_path=FLAGS.kalign_binary_path,
        release_dates_path=None,
        obsolete_pdbs_path=FLAGS.obsolete_pdbs_path,
        custom_tempdir=FLAGS.custom_tempdir,
        strict_error_check=True)
  else:
    template_searcher = hhsearch.HHSearch(
        binary_path=FLAGS.hhsearch_binary_path,
        hhalign_binary_path=FLAGS.hhalign_binary_path,
        databases=[FLAGS.pdb70_database_path],
        custom_tempdir=FLAGS.custom_tempdir)
    template_featurizer = templates.HhsearchHitFeaturizer(
        mmcif_dir=FLAGS.template_mmcif_dir,
        max_template_date=FLAGS.max_template_date,
        max_hits=MAX_TEMPLATE_HITS,
        kalign_binary_path=FLAGS.kalign_binary_path,
        release_dates_path=None,
        obsolete_pdbs_path=FLAGS.obsolete_pdbs_path,
        custom_tempdir=FLAGS.custom_tempdir,
        strict_error_check=True)

  monomer_data_pipeline = pipeline.DataPipeline(
      jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
      hhblits_binary_path=FLAGS.hhblits_binary_path,
      uniref90_database_path=FLAGS.uniref90_database_path,
      mgnify_database_path=FLAGS.mgnify_database_path,
      bfd_database_path=FLAGS.bfd_database_path,
      uniclust30_database_path=FLAGS.uniclust30_database_path,
      small_bfd_database_path=FLAGS.small_bfd_database_path,
      template_searcher=template_searcher,
      template_featurizer=template_featurizer,
      use_small_bfd=use_small_bfd,
      use_precomputed_msas=FLAGS.use_precomputed_msas,
      custom_tempdir=FLAGS.custom_tempdir)

  if run_multimer_system:
    num_predictions_per_model = FLAGS.num_multimer_predictions_per_model
    data_pipeline = pipeline_multimer.DataPipeline(
        monomer_data_pipeline=monomer_data_pipeline,
        jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
        uniprot_database_path=FLAGS.uniprot_database_path,
        use_precomputed_msas=FLAGS.use_precomputed_msas,
        custom_tempdir=FLAGS.custom_tempdir)
  else:
    num_predictions_per_model = 1
    data_pipeline = monomer_data_pipeline

  model_runners = {}
  model_names = config.MODEL_PRESETS[FLAGS.model_preset]
  for model_name in model_names:
    model_config = config.model_config(model_name, FLAGS.num_recycle)
    if run_multimer_system:
      model_config.model.num_ensemble_eval = num_ensemble
    else:
      model_config.data.eval.num_ensemble = num_ensemble
    model_params = data.get_model_haiku_params(
        model_name=model_name, data_dir=FLAGS.data_dir)
    model_runner = model.RunModel(model_config, model_params)
    for i in range(num_predictions_per_model):
      model_runners[f'{model_name}_pred_{i}'] = model_runner

  logging.info('Have %d models: %s', len(model_runners),
               list(model_runners.keys()))

  if FLAGS.run_relax:
    amber_relaxer = relax.AmberRelaxation(
        max_iterations=RELAX_MAX_ITERATIONS,
        tolerance=RELAX_ENERGY_TOLERANCE,
        stiffness=RELAX_STIFFNESS,
        exclude_residues=RELAX_EXCLUDE_RESIDUES,
        max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS,
        use_gpu=FLAGS.use_gpu_relax)
  else:
    amber_relaxer = None

  random_seed = FLAGS.random_seed
  if random_seed is None:
    random_seed = random.randrange(sys.maxsize // len(model_runners))
  logging.info('Using random seed %d for the data pipeline', random_seed)



  #Code adaptions to handle custom template, no MSA, no template
  #Check that no_msa_list has same number of elements as in fasta_sequence,
  #and convert to bool.
  parsed_sequences = parse_fasta(FLAGS.fasta_path)
  if FLAGS.no_msa_list:
    if len(FLAGS.no_msa_list) != len(parsed_sequences):
      raise ValueError('--no_msa_list must either be omitted or match '
                       'number of sequences.')
    no_msa_list = []
    for s in FLAGS.no_msa_list:
      if s.lower() == 'true':
        no_msa_list.append(True)
      elif s.lower() == 'false':
        no_msa_list.append(False)
      else:
        raise ValueError('--no_msa_list must contain comma separated '
                         'true or false values.')
  else:
    no_msa_list = [False] * len(parsed_sequences)


  if FLAGS.custom_template_list:
      if len(FLAGS.custom_template_list) != len(parsed_sequences):
          raise ValueError('--custom_template_list must either be omitted or match '
                           'number of sequences.')
      custom_template_list = []
      for s in FLAGS.custom_template_list:
          if s in ["None", "none"]:
            custom_template_list.append(None)
          else:
            custom_template_list.append(s)

  else:
      custom_template_list = [None] * len(parsed_sequences)

  if FLAGS.no_template_list:
      if len(FLAGS.no_template_list) != len(parsed_sequences):
          raise ValueError('--no_template_list must either be omitted or match '
                           'number of sequences.')
      no_template_list = []
      for s in FLAGS.no_template_list:
          if s.lower() == 'true':
              no_template_list.append(True)
          elif s.lower() == 'false':
              no_template_list.append(False)
          else:
              raise ValueError('--no_template_list must contain comma separated '
                               'true or false values.')
  else:
      no_template_list = [False] * len(parsed_sequences)

  predict_structure(
        fasta_path=FLAGS.fasta_path,
        fasta_name=fasta_name,
        output_dir_base=FLAGS.output_dir,
        data_pipeline=data_pipeline,
        model_runners=model_runners,
        amber_relaxer=amber_relaxer,
        benchmark=FLAGS.benchmark,
        random_seed=random_seed,
        no_msa_list=no_msa_list,
        no_template_list=no_template_list,
        custom_template_list=custom_template_list)

def sigterm_handler(_signo, _stack_frame):
    raise KeyboardInterrupt

if __name__ == '__main__':
  flags.mark_flags_as_required([
      'fasta_path',
      'output_dir',
      'data_dir',
      'uniref90_database_path',
      'mgnify_database_path',
      'template_mmcif_dir',
      'max_template_date',
      'obsolete_pdbs_path',
      'use_gpu_relax',
  ])
  try:
    import signal
    signal.signal(signal.SIGTERM, sigterm_handler)
    app.run(main)
  except KeyboardInterrupt:
      logging.error("Alphafold pipeline was aborted. Exit code 2")
  except Exception:
      logging.info(traceback.print_exc())
      logging.error("Alphafold pipeline finished with an error. Exit code 1")
